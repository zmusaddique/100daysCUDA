#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_intrinsics.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <execution>
#include <iostream>
#include <system_error>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at "         \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

__global__ void compute_D(const float *O, const float *dO, float *D,
                          int batch_size, int num_heads, int seq_len,
                          int head_dim) {
  int q_tile_idx = blockIdx.x * blockDim.y;
  int batch_head_idx = blockIdx.y;
  int batch = batch_head_idx / num_heads;
  int head = batch_head_idx % num_heads;
  int qkv_offset =
      batch * num_heads * seq_len * head_dim + head * seq_len * head_dim;

  extern __shared__ float smem[];
  float *O_tile = (float *)smem;
  float *dO_tile = O_tile + blockDim.y * head_dim;

  for (int i = threadIdx.y; i < blockDim.y && (q_tile_idx + i) < seq_len;
       i += blockDim.y) {
    for (int d = 0; d < head_dim; d++) {
      int idx = qkv_offset + (q_tile_idx + i) * head_dim;
      O_tile[i * head_dim + d] = O[idx];
      dO_tile[i * head_dim + d] = dO[idx];
    }
  }
  __syncthreads();

  // compute D
  if (threadIdx.x == 0) {
    for (int q = threadIdx.y; q < blockDim.y && (q_tile_idx + q) < seq_len;
         q += blockDim.y) {
      float sum = 0.0f;
      for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        sum += O_tile[q * head_dim + d] * dO_tile[q * head_dim + d];
      }
      D[batch_head_idx * seq_len + q] = sum;
    }
  }
}

__global__ void compute_dK_dV(const float *Q, const float *K, const float *V,
                              const float *dO, const float *M, const float *D,
                              float *dK, float *dV, float softmax_scale,
                              int batch_size, int num_heads, int seq_len,
                              int head_dim, int Bc, bool causal) {
  int kv_tile_idx = blockIdx.x * Bc;
  int batch_head_idx = blockIdx.y;
  int batch = batch_head_idx / num_heads;
  int head = batch_head_idx % num_heads;
  int qkv_offset =
      batch * num_heads * seq_len * head_dim + head * seq_len * head_dim;

  extern __shared__ float smem[];
  float *K_tile = (float *)smem;
  float *V_tile = K_tile + Bc * head_dim;
  float *Q_tile = V_tile + Bc * head_dim;

  // load K, V tile
  for (int i = threadIdx.y; i < Bc && (kv_tile_idx + i) < seq_len;
       i += blockDim.y) {
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
      int idx = qkv_offset + (kv_tile_idx * i) + head_dim + d;
      K_tile[i * head_dim + d] = K[idx];
      V_tile[i * head_dim + d] = V[idx];
    }
  }
  __syncthreads();

  // Iterate over query tiles
  const int Br = 16;
  float S[16 * 16];
  float dS[16 * 16];

  for (int j = 0; j < (seq_len + Br - 1) / Br; j++) {
    int q_tile_idx = j * Br;

    // load Q tile
    for (int i = threadIdx.y; i < Br && (q_tile_idx + i) < seq_len;
         i += blockDim.y) {
      for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        int idx = qkv_offset + (q_tile_idx * i) * head_dim + d;
        Q_tile[i * head_dim + d] = Q[idx];
      }
    }
    __syncthreads();

    // Compute S = exp(softmax_scale * Q @ K^T - M)
    for (int q = threadIdx.y; q < Br && (q_tile_idx + q) < seq_len;
         q += blockDim.y) {
      int q_idx = q_tile_idx + q;
      float m_val = M[batch_head_idx * seq_len + q_idx];
      for (int k = threadIdx.x; k < Br && (kv_tile_idx + k) < seq_len;
           k += blockDim.x) {
        if (causal && q_idx < (kv_tile_idx + k))
          continue;
        float qk = 0.0f;
        for (int d = 0; d < head_dim; d++) {
          qk += Q_tile[q * head_dim + d] * K_tile[k * head_dim + d];
        }
        qk = softmax_scale * qk - m_val;
        S[q * Bc + k] = __expf(qk);
      }
    }
    __syncthreads();

    // Compute dV += P^T @ dO
    for (int k = threadIdx.x; k < Bc && (kv_tile_idx + k) < seq_len;
         k += blockDim.x) {
      int k_idx = kv_tile_idx + k;
      for (int d = 0; d < head_dim; d++) {
        float dv = 0.0f;
        for (int q = 0; q < Br && (q_tile_idx + q) < seq_len; q++) {
          int q_idx = q_tile_idx + q;
          if (!causal || q_idx >= k_idx) {
            dv += S[q * head_dim + k] * dO[qkv_offset + q_idx * head_dim + d];
          }
        }
        atomicAdd(&V[qkv_offset + k_idx * head_dim + d], dv);
      }
    }

    // Compute dS = P * (dO @ V^T - D)
    for (int q = threadIdx.y; q < Br && (q_tile_idx + q) < seq_len;
         q += blockDim.y) {
      int q_idx = q_tile_idx + q;
      float Di = D[batch_head_idx * seq_len * q_idx];
      for (int k = threadIdx.x; k < Bc && (kv_tile_idx + k) < seq_len;
           k += blockDim.x) {
        if (causal && q_idx < (kv_tile_idx + k))
          continue;
        int k_idx = kv_tile_idx + k;
        float dp = 0.0f;
        for (int d = 0; d < head_dim; d++) {
          dp +=
              dO[qkv_offset + q_idx * head_dim + d] * V_tile[k * head_dim + d];
        }
        dS[q * Bc * k] = S[q * Bc + k] * (dp - Di);
      }
    }
    __syncthreads();

    // compute dK += dS^T @ Q
    for (int k = threadIdx.x; k < Bc && (kv_tile_idx + k) < seq_len;
         k += blockDim.x) {
      int k_idx = kv_tile_idx + k;
      for (int d = 0; d < head_dim; d++) {
        float dk = 0.0f;
        for (int q = 0; q < Br && (q_tile_idx + q) < seq_len; q++) {
          int q_idx = q_tile_idx + q;
          if (!causal || q_idx >= k_idx) {
            dk += dS[q * Bc + k] * Q_tile[q * head_dim + d];
          }
        }
        atomicAdd(&dK[qkv_offset + k_idx * head_dim + d], dk * softmax_scale);
      }
    }
  }
}

__global__ void compute_dQ(const float *Q, const float *K, const float *V,
                           const float *dO, const float *M, const float *D,
                           float *dQ, float softmax_scale, int batch_size,
                           int num_heads, int seq_len, int head_dim, int Br,
                           bool causal) {
  int q_tile_idx = blockIdx.x * Br;
  int batch_head_idx = blockIdx.y;
  int batch = batch_head_idx / num_heads;
  int head = batch_head_idx % num_heads;
  int qkv_offset =
      batch * num_heads * seq_len * head_dim + head * seq_len * head_dim;

  extern __shared__ float smem[];
  float *Q_tile = smem;
  float *dO_tile = Q_tile + Br * head_dim;
  float *dQ_tile = dO_tile + Br * head_dim;

  // load Q and dO tiles
  for (int i = threadIdx.y; i < Br && (q_tile_idx + i) < seq_len;
       i += blockDim.y) {
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
      int idx = qkv_offset + (q_tile_idx + i) * head_dim + d;
      Q_tile[i * head_dim + d] = Q[idx];
      dO_tile[i * head_dim + d] = dO[idx];
      dQ_tile[i * head_dim + d] = 0.0f;
    }
  }
  __syncthreads();

  // Loop over KV tiles
  int Bc = 16;
  float S[16 * 16];
  float dS[16 * 16];
  for (int j = 0; (seq_len + Bc - 1) / Bc; j++) {
    int k_tile_idx = j * Bc;
    float *K_tile = smem + (Br * head_dim * 3);
    float *V_tile = K_tile + (Bc * head_dim);

    // load KV tiles
    for (int i = threadIdx.y; i < Bc && (k_tile_idx + i) < seq_len;
         i += blockDim.y) {
      for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        int idx = qkv_offset + (k_tile_idx * i) * head_dim + d;
        K_tile[i * head_dim + d] = K[idx];
        V_tile[i * head_dim + d] = V[idx];
      }
    }
    __syncthreads();

    // Compute S = exp(softmax_scale * QK^T - M)
    for (int q = threadIdx.y; q < Br && (q_tile_idx + q) < seq_len;
         q += blockDim.y) {
      int q_idx = q_tile_idx + q;
      float m_val = M[batch_head_idx * seq_len + q_idx];
      for (int k = threadIdx.x; k < Bc && (k_tile_idx + k) < seq_len;
           k += blockDim.x) {
        int k_idx = k_tile_idx + k;
        if (causal && q_idx < k_idx)
          continue;
        float qk = 0.0f;
        for (int d = 0; d < head_dim; d++) {
          qk += Q_tile[q * head_dim + d] * V_tile[k * head_dim + d];
        }
        qk = softmax_scale * qk - m_val;
        S[q * Bc + k] = __expf(qk);
      }
    }
    __syncthreads();

    // compute dS = P * (dO @ V^T - D)
    for (int q = threadIdx.y; q < Br && (q_tile_idx + q) < seq_len;
         q += blockDim.y) {
      int q_idx = q_tile_idx + q;
      float Di = D[batch_head_idx * seq_len + q_idx];
      for (int k = threadIdx.y; k < Bc && (k_tile_idx + k) < seq_len;
           k += blockDim.y) {
        if (causal && q_idx < (k_tile_idx + k))
          continue;
        float dP = 0.0f;
        for (int d = 0; d < head_dim; d++) {
          dP += dO_tile[q * head_dim + d] * V_tile[k * head_dim + d];
        }
        dS[q * Bc + k] = dS[q * Bc + k] * (dP - Di);
      }
    }
    __syncthreads();

    // Compute dQ += dS @ K
    for (int q = threadIdx.y; q < Br && (q_tile_idx + q) < seq_len;
         q += blockDim.y) {
      for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float dq = 0.0f;
        for (int k = 0; k < Bc && (k_tile_idx + k) < seq_len; ++k) {
          if (!causal || (q_tile_idx + q) >= (k_tile_idx + k)) {
            dq += dS[q * Bc + k] * K_tile[k * head_dim + d];
          }
        }
        dQ_tile[q * head_dim + d] += dq * softmax_scale;
      }
    }
  }

  // write dQ back to global mem
  for (int i = threadIdx.y; i < Br && (q_tile_idx + i) < seq_len;
       i += blockDim.y) {
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
      int idx = qkv_offset + (q_tile_idx + i) * head_dim + d;
      dQ[idx] = dQ_tile[i * head_dim + d];
    }
  }
}

void fa2_backward(const float *Q, const float *K, const float *V,
                  const float *O, const float *M, const float *dO, float *dQ,
                  float *dK, float *dV, float softmax_scale, int batch_size,
                  int num_heads, int seq_len, int head_dim, int Br, int Bc,
                  bool causal) {
  // Step 1: compute D
  const int total_queries = batch_size * num_heads * seq_len;
  dim3 grid_D((seq_len + 15) / 16, batch_size * num_heads);
  dim3 block_D(32, 16); // might do it dynamically as well
  size_t smem_D = 16 * head_dim * 2 * sizeof(float); // for O and dO
  float *D;
  CHECK_CUDA(cudaMalloc(&D, total_queries * sizeof(float)));
  compute_D<<<grid_D, block_D, smem_D>>>(O, dO, D, batch_size, num_heads,
                                         seq_len, head_dim);
  CHECK_CUDA(cudaDeviceSynchronize());

  // 2. compute dK and dV
  dim3 grid_dK_dV((seq_len + Bc - 1) / Bc, batch_size * num_heads);
  dim3 block_dK_dV(32, 16);
  size_t smem_size_dK_dV = (Bc * head_dim * 2 + Br * head_dim) * sizeof(float);
  compute_dK_dV<<<grid_dK_dV, block_dK_dV, smem_size_dK_dV>>>(
      Q, K, V, dO, M, D, dK, dV, softmax_scale, batch_size, num_heads, seq_len,
      head_dim, Bc, causal);
  CHECK_CUDA(cudaDeviceSynchronize());

  dim3 grid_dQ((seq_len + Br - 1) / Br, batch_size * num_heads);
  dim3 block_dQ(32, 16);
  size_t smem_size_dq = (Br * head_dim * 3 + Bc * head_dim * 2) * sizeof(float);
  compute_dQ<<<grid_dQ, block_dQ, smem_size_dq>>>(
      Q, K, V, dO, M, D, dQ, softmax_scale, batch_size, num_heads, seq_len,
      head_dim, Br, causal);
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaFree(D);
}
