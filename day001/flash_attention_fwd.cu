// This is Flash attention forward pass
#include <stdio.h>

#define CUDA_CHECK(ans)                 \
  {                                     \
  cudaAssert((ans), __FILE__, __LINE__);\
  }
inline void cudaAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA error %s: %s at %s: %d\n", cudaGetErrorName(code),
            cudaGetErrorString(code), file, line);
    exit(code);
  }
}

template <const int Br, const int Bc>
__global__ void flash_attn_kernel(
  float *Q,
  float *K, 
  float *V, 
  int N, int d, 
  int Tr, int Tc, 
  float scale, 
  float *L, float *O
) {
  extern __shared__ float smem[];
  // We will partition this shared mem into diff segments

  float *Q_i = smem; // First seg used for Query
  float *K_j = Q_i + (Br * d); // Plus offset of Q_i
  float *V_j = K_j + (Bc * d); // Plus size of K_i
  float *S_ij = V_j + (Bc * d); // Plus size of V_j -> This is attention scores
  float *O_i = S_ij + (Br * Bc); // Plus size of attention scores, S
  // O_i size should be Br * d
  float *li = O_i + (Br * d); // Row sums : Br elements => 1/block
  float *mi = li + Br; // row maxima of attention : Br elements
  float *mi_new = mi + Br; // updated row maxima: Br elements

  
  int tx = threadIdx.x; // 0 <= tx < Br * Bc
  int srow = tx / Bc; // row idx in tile
  int scol = tx % Bc; // col idx in tile

  int bx = blockIdx.x; // batch idx
  int by = blockIdx.y; // head idx

  // Offset: How many elements to skip?
  int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d); // block + head
  int lm_offset = (bx * gridDim.y * N) + (by * N);

  for (int i = 0; i < Tr; i++ ){
    // Load Qi on sram
    for (int e = tx; e < Br * d; e += Br * Bc){
      int row = e / d;
      int col = e % d;
      if ((i * Br + row) < N) {
        Q_i[row * d + col] = Q[qkv_offset + (i * Br + row) * d + col];
        O_i[row * d + col] = 0.0f;
      }
    }

    if (scol == 0) {
      li[srow] = 0.0f;
      mi[srow] = -INFINITY;
      mi_new[srow] = -INFINITY;
    }

    __syncthreads();
  
    // Attention score computation
    for (int j = 0; j < Tc; j++) {
      // Load Kj and Vj
      for (int e = tx; e < Bc * d; e += Br * Bc) {
        int row = e / d;
        int col = e % d;
        if ((j * Bc + row )< N) {
          K_j[row * d + col] = K[qkv_offset + (j * Bc + row) * d + col];
          V_j[row * d + col] = V[qkv_offset + (j * Bc + row) * d + col];
        }
      }
      __syncthreads();
      
      // Compute S_ij = Qi x Kj^T
      float score = 0.0f;
      for (int k = 0; k < d; k++) {
        score += Q_i[srow * d + k] * K_j[scol * d + k]; 
      }
      score *= scale;

      S_ij[srow * d + scol] = score;
      __syncthreads();


      // row statistics
      if (scol == 0) {
        float tile_max = -INFINITY;
        for (int c = 0; c < Bc; c++) {
          tile_max = fmaxf(tile_max, S_ij[srow * Bc + c]);
        }

        float m_old = mi[srow];
        float m_new_val=  fmaxf(m_old, tile_max);

        // Normaliztion factor
        float tile_sum = 0.0f;
        for (int c = 0; c < Bc; c++) {
          float exp_val = expf(S_ij[srow * Bc + c] - m_new_val);
          tile_sum += exp_val;

          S_ij[srow * Bc + c] = exp_val;
        }
        li[srow] = li[srow] * expf(m_old - m_new_val) + tile_sum;
        mi_new[srow] = m_old;
        mi[srow] = m_new_val;
      }
      __syncthreads();

      float m_old_val = mi_new[srow];
      for (int col = scol; col < d; col += Bc) {
        float weighted_sum = 0.0f;

        for (int c = 0; c < Bc; c++) {
          weighted_sum += S_ij[srow * Bc + c] * V_j[c * d + col]; 
        }

        O_i[srow * d + scol] = O_i[srow * d + col] * expf(m_old_val - mi[srow]) + weighted_sum;
      }
      __syncthreads();
    } // end key tile loop

    // final normalization of O_i
    for (int col = scol; col < d; col++) {
      O_i[srow * d + col] /= li[srow];
    }
    __syncthreads();

    // Write this tile back to global memory
    for (int e = tx; e < Br * d; e += Br * Bc){
      int row = e / d;
      int col = e % d;
      if ((i * Br + row) < N){
        O[qkv_offset + (i * Br + row) * d + col] = O_i[row * d + col];
      }
    }

    // log-scale normalization 
    if (scol == 0) {
      // log(li) + current_max
      L[lm_offset + i * Br + srow] = logf(li[srow]) + mi[srow];
    }
    __syncthreads();
  }
}


int main() {
  int batch_size = 16;
  int n_heads = 8;
  int seq_len = 512;
  int head_dim = 64;

  /* Processing a batch with each head (having head_dim) covering different
    parts of a sequence of length seq_len
  */
  int Q_size = batch_size * n_heads * seq_len * head_dim;
  int K_size = batch_size * n_heads * seq_len * head_dim;
  int V_size = batch_size * n_heads * seq_len * head_dim;

  int lm_size = batch_size * n_heads * seq_len;

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
  float softmax_scale = 1.0 / sqrt(head_dim);

  /*
  Br * Bc: Attention scores tile (S_tile).
  2 * Br * head_embd: Query tile (Q_tile) and output tile (O_tile).
  2 * Bc * head_embd: Key tile (K_tile) and value tile (V_tile).
  3 * Br: Statistics arrays (m_tile, l_tile, m_prev_tile).
  */
  const int smem_size =
      ((Br * Bc) + (2 * Bc * head_dim) + (2 * Br * head_dim) + (3 * Br)) *
      sizeof(float);
  int max_sram_size;
  cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
  printf("Max shared memory in device: %d\nRequested shared memory: %d\n",
         max_sram_size, smem_size);

  dim3 grid(batch_size, n_heads);
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
  CUDA_CHECK(cudaMemcpy(Q, Q_h, Q_size * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(K, K_h, K_size * sizeof(float), cudaMemcpyHostToDevice)); 
  CUDA_CHECK(cudaMemcpy(V, V_h, V_size * sizeof(float), cudaMemcpyHostToDevice));  
  CUDA_CHECK(cudaMemcpy(O, O_h, Q_size * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(L, L_h, lm_size * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  printf("Host to device copy time: %f\n", ms);

  flash_attn_kernel<Br, Bc><<<grid, block, smem_size>>>(Q, K, V, seq_len, head_dim, Tr, Tc, softmax_scale, L, O);
}
