#include <stdio.h>
#include <torch/extension.h>

#define CUDA_CHECK(ans)                                                        \
  {                                                                            \
    cudaAssert((ans), __FILE__, __LINE__);                                     \
  }
inline void cudaAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA error %s: %s torch %s: %d\n", cudaGetErrorName(code),
            cudaGetErrorString(code), file, line);
    exit(code);
  }
}

template <const int Br, const int Bc>
__global__ void flash_attention_fwd(float *Q, float *K, float *V, float *O,
                                    float *L, int N, int d, int Tr, int Tc) {
  extern __shared__ float smem[];

  float *Q_i = smem;
  float *K_j = Q_i + (Br * d);
  float *V_j = K_j + (Bc * d);
  float *S_ij = V_j + (Bc * d);
  float *O_i = S_ij + (Br * Bc);

  float *l_i = O_i + (Br * d);
  float *m_i = l_i + Br;
  float *m_old_i = m_i + Br;
  float *L_i = m_old_i + Br;

  int batch_idx = blockIdx.x;
  int head_idx = blockIdx.y;
  int tx = threadIdx.x;

  int s_row = tx / Bc;
  int s_col = tx % Bc;

  int qkv_offset = (batch_idx * gridDim.y * N * d) + (head_idx * N * d);
  for (int i = 0; i < Tr; i++) {
    for (int e = tx; e < Br * d; e += Br * Bc) {
      int row = e / d;
      int col = e % d;
      if ((i * Br + row) < N) {
        Q_i[row * d + col] = Q[qkv_offset + (i * Br + row) * d + col];
        O_i[row * d + col] = 0.0f;
      }
    }

    if (s_col == 0) {
      l_i[s_row] = 0.0f;
      m_i[s_row] = -INFINITY;
    }

    __syncthreads();

    for (int j = 0; j < Tc; j++) {
      for (int e = tx; e < Bc * d; e += Br * Bc) {
        int row = e / d;
        int col = e % d;
        if ((j * Bc + row) < N) {
          K_j[row * d + col] = K[qkv_offset + (j * Bc + row) * d + col];
          V_j[row * d + col] = V[qkv_offset + (j * Bc + row) * d + col];
        }
      }
      __syncthreads();

      float acc = 0.0f;
      for (int k = 0; k < d; k++) {
        acc += Q_i[s_row * d + k] * K_j[s_col * d + k];
      }
      S_ij[s_row * Bc + s_col] = acc;
      __syncthreads();

      if (s_col == 0) {
        float tile_max = -INFINITY;
        for (int c = 0; c < Bc; c++) {
          tile_max = fmaxf(tile_max, S_ij[s_row * Bc + c]);
        }

        float m_old = m_i[s_row];
        m_old_i[s_row] = m_old;
        float m_new = fmax(tile_max, m_old);
        m_i[s_row] = m_new;

        float sum_P = 0.0f;
        for (int c = 0; c < Bc; c++) {
          S_ij[s_row * Bc + c] = expf(S_ij[s_row * Bc + c] - m_new); // P_ij
          sum_P += S_ij[s_row * Bc + c];
        }

        l_i[s_row] = expf(m_old - m_new) * l_i[s_row] + sum_P;
      }
      __syncthreads();

      for (int col = s_col; col < d; col += Bc) {
        if (col < d) {
          float scale = expf(m_old_i[s_row] - m_i[s_row]);
          O_i[s_row * d + col] *= scale;

          float sum = 0.0f;
          for (int k = 0; k < Bc; k++) {
            sum += S_ij[s_row * Bc + k] * V_j[k * d + col];
          }
          O_i[s_row * d + col] += sum;
        }
      }
      __syncthreads();
    }
    for (int e = tx; e < Br * d; e += blockDim.x) {
      int row = e / d;
      int col = e % d;
      if ((i * Br + row) < N) {
        O_i[row * d + col] /= l_i[row];
      }
    }

    if (tx < Br && (i * Br + tx) < N) {
      L_i[tx] = m_i[tx] + logf(l_i[tx]);
    }

    for (int e = tx; e < Br * d; e += blockDim.x) {
      int row = e / d;
      int col = e % d;
      if ((i * Br + row) < N) {
        O[qkv_offset + (i * Br + row) * d + col] = O_i[row * d + col];
      }
    }

    if (tx < Br && (i * Br + tx) < N) {
      L[qkv_offset / d + (i * Br + tx)] = L_i[tx];
    }
  }
}

torch::Tensor fa2_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {

  const int Bc = 16;
  const int Br = 16;

  int B = Q.size(0);
  int h = Q.size(1);
  int N = Q.size(2);
  int d = Q.size(3);

  int Tc = ceil((float)N / Bc);
  int Tr = ceil((float)N / Br);
  float softmax_scale = 1.0 / sqrt(d);

  auto O = torch::zeros_like(Q);
  auto L = torch::zeros({B, h, N});
  torch::Device device(torch::kCUDA);
  L = L.to(device);

  int smem_size = 2 * (Br * d) + 2 * (Bc * d) + (Br * Bc) + (3 * Br);

  int max_sram_size;
  cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
  printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size,
         smem_size);

  dim3 grid_size(B, h);     // batch_size x num_heads
  dim3 block_size(Br * Bc); // Br * Bc threads per block

  flash_attention_fwd<Br, Bc><<<grid_size, block_size, smem_size>>>(
      Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
      O.data_ptr<float>(), L.data_ptr<float>(), N, h, Tr, Tc);
  return O;
}

int main() {
  int batch_size = 16;
  int num_heads = 8;
  int seq_len = 512;
  int head_dim = 64;

  int lm_size = batch_size * num_heads * seq_len;

  int Q_size = batch_size * num_heads * seq_len * head_dim;
  int K_size = batch_size * num_heads * seq_len * head_dim;
  int V_size = batch_size * num_heads * seq_len * head_dim;

  float *Q_h = (float *)malloc(Q_size * sizeof(float));
  float *K_h = (float *)malloc(K_size * sizeof(float));
  float *V_h = (float *)malloc(V_size * sizeof(float));
  float *O_h = (float *)malloc(Q_size * sizeof(float));
  float *L_h = (float *)malloc(lm_size * sizeof(float));

  for (int i = 0; i < Q_size; i++) {
    Q_h[i] = 1.0f;
    K_h[i] = 2.0f;
    V_h[i] = 3.0f;
    O_h[i] = 0.0f;
  }

  for (int i = 0; i < lm_size; i++) {
    L_h[i] = 0.0f;
  }

  const int Br = 16, Bc = 16;
  int Tr = ceil((float)seq_len / Br);
  int Tc = ceil((float)seq_len / Bc);
  int smem_size =
      2 * (Br * head_dim) + 2 * (Bc * head_dim) + (Br * Bc) + (3 * Br);

  int max_sram_size;
  cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
  printf("Max shared memory in device: %d\nRequested shared memory: %d\n",
         max_sram_size, smem_size);

  dim3 grid(batch_size, num_heads);
  dim3 block(Br * Bc); // Simplicity for now :)

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  float ms = 0.0f;

  float *Q, *K, *V, *O, *L;

  CUDA_CHECK(cudaEventRecord(start));
  CUDA_CHECK(cudaMalloc(&Q, Q_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&K, K_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&V, V_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&O, Q_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&L, lm_size * sizeof(float)));
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  printf("GPU memory allocation time: %f\n", ms);

  cudaEventRecord(start);
  CUDA_CHECK(
      cudaMemcpy(Q, Q_h, Q_size * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(K, K_h, K_size * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(V, V_h, V_size * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(O, O_h, Q_size * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(L, L_h, lm_size * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  printf("Host to device copy time: %f\n", ms);

  cudaEventRecord(start);
  flash_attention_fwd<Br, Bc>
      <<<grid, block, smem_size>>>(Q, K, V, O, L, seq_len, head_dim, Tr, Tc);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);
  printf("Flash attention kernel execution time: %f ms\n", ms);

  cudaEventRecord(start);
  cudaMemcpy(O_h, O, Q_size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);
  printf("Time to copy output from device to host %f ms\n", ms);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  printf("First and last value in Output: \n%f and %f\n", O_h[0],
         O_h[Q_size - 1]);

  cudaFree(Q);
  cudaFree(K);
  cudaFree(V);
  cudaFree(O);
  free(Q_h);
  free(K_h);
  free(V_h);
  free(O_h);
}
