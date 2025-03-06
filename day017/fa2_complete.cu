#include <__clang_cuda_builtin_vars.h>
#include <iostream>
#include <torch/extension.h>
#include <tuple>

// Error checking macro
#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at "         \
                << __FILE__ << ":" << __LINE__ << " Call: " << #call           \
                << std::endl;                                                  \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

__global__ void fa2_fwd(float *Q, float *K, float *V, float *O, float *M,
                        float *L, float *S_debug, float *P_debug,
                        float softmax_scale, int num_heads, int batch_size,
                        int seq_len, int head_dim, int Br, int Bc, int Tr,
                        int Tc, bool causal) {
  const int ty = threadIdx.y;
  const int tx = threadIdx.x;

  const int block_idx_q = blockIdx.x; // tile ID
  const int batch_head_idx = blockIdx.y;
  const int batch_idx = batch_head_idx / num_heads;
  const int head_idx = batch_head_idx % num_heads;

  const int q_tile_idx = blockIdx.x * Br;

  // i * stride
  const int batch_offset = batch_idx * (num_heads * seq_len * head_dim);
  const int head_offset = head_idx * (seq_len * head_dim);
  // qkv_offset is used to skip the sequences and head dimensions from the
  // starting pointed loc
  const int qkv_offset = batch_offset + head_offset;

  extern __shared__ float smem[];

  const int head_dim_padded = head_dim + (head_dim % 32 == 0 ? 1 : 0);
  const int Bc_padded = Bc + (Bc % 32 ? 1 : 0);

  float *Q_tile = (float *)smem;
  float *K_tile = Q_tile + (Br * head_dim_padded);
  float *V_tile = K_tile + (Bc * head_dim_padded);
  float *l = V_tile + (Bc * head_dim_padded);
  float *m = l + Br;
  float *o = m + Br;
  float *qk = o + (Br * head_dim_padded);

  for (int e = tx; e < Bc_padded; e += blockDim.x) {
    if (e < Bc)
      qk[e] = 0.0f;
  }
  __syncthreads();

// Load Q tile
#pragma unroll 4
  for (int i = ty; i < Br; i += blockDim.y) {
    int row = q_tile_idx + i;
    const bool valid_row = row < seq_len;

    const int base_idx = qkv_offset + row * head_dim;
#pragma unroll 4
    for (int j = tx; j < head_dim; j += blockDim.x) {
      Q_tile[i * head_dim_padded + j] = valid_row ? Q[base_idx + j] : 0.0f;
    }
  }

  // Accumulators
  if (tx == 0 && ty < Br) {
    m[ty] = -INFINITY;
    l[ty] = 0.0f;
  }

#pragma unroll 4
  for (int c = tx; c < head_dim; c += blockDim.x) {
    o[ty * head_dim_padded + c] = 0.0f;
  }
  __syncthreads();

  // Looping over KV_tiles
  for (int j = 0; j < Tc; j++) {
    int j_start = j * Bc;
// Load K, V
#pragma unroll 4
    for (int i = ty; i < Bc; i += blockDim.y) {
      const int row = j_start + i;
      const bool valid_row = row < seq_len;
      const int base_idx = qkv_offset + row * head_dim;

      for (int c = tx; c < head_dim; c += blockDim.x) {
        if (valid_row) {
          K_tile[i * head_dim_padded + c] = K[base_idx + c];
          V_tile[i * head_dim_padded + c] = V[base_idx + c];
        } else {
          K_tile[i * head_dim_padded + c] = 0.0f;
          V_tile[i * head_dim_padded + c] = 0.0f;
        }
      }
    }

    // Resetting qk for each new kv tile
    const int row_offset = ty * Bc_padded;

#pragma unroll 4
    for (int k = 0; k < Bc; k++) {
      qk[row_offset + k] = 0.0f;
    }

    __syncthreads();

// QK^T
#pragma unroll 4
    for (int d = 0; d < head_dim; d++) {
      const float qval = Q_tile[threadIdx.y * head_dim_padded + d];
      for (int k = 0; k < Bc; k++) {
        qk[row_offset + k] += qval * K_tile[k * head_dim_padded + d];
      }
    }

    // Apply softmax: causal mask and scale in one pass
    float max_val = -INFINITY;
    const int q_global_idx = q_tile_idx + ty;

#pragma unroll 4
    for (int k = 0; k < Bc; k++) {
      const int k_global_idx = j_start + k;
      float val = qk[row_offset + k];

      if (causal && q_global_idx < k_global_idx) {
        qk[row_offset + k] = -INFINITY;
      } else {
        val *= softmax_scale;
        max_val = fmaxf(max_val, val);
        qk[row_offset + k] = val;
      }
      if (k_global_idx < seq_len && q_global_idx < seq_len) {
        S_debug[(batch_head_idx * seq_len + q_global_idx) * seq_len +
                k_global_idx] = __expf(val - max_val);
      }
    }

    // warp-level reduction for max value
    for (int offset = 0; offset > 0; offset /= 2) {
      float other = __shfl_down_sync(0xffffffff, max_val, offset);
      max_val = fmaxf(max_val, other); // thread 0 will have the max-val
    }

    // Broadcast to other threads in warp
    max_val = __shfl_sync(0xffffffff, max_val, 0); // broadcast from t0 to all

    float m_prev = m[ty];
    float m_new = fmaxf(m_prev, max_val);
    float exp_scale = __expf(m_prev - m_new);

    float l_prev = l[ty] * exp_scale;
    float l_new = 0.0f;

#pragma unroll 4
    for (int k = 0; k < Bc; k++) {
      l_new += __expf(qk[row_offset + k] - m_new);
    }

    const float alpha = expf(m_prev - m_new);
#pragma unroll 4
    for (int d = 0; d < head_dim; d++) {
      float acc = o[ty * head_dim_padded + d] * exp_scale;

#pragma unroll 4
      for (int k = 0; k < Bc; k++) {
        const float p = __expf(qk[row_offset + k] - m_new);
        acc += p * V_tile[k * head_dim_padded + d];
        if (j_start + k < seq_len && q_global_idx < seq_len) {
          P_debug[(batch_head_idx * seq_len + q_global_idx) * seq_len +
                  (j_start + k)] = p / (l_prev + l_new);
        }
      }
      o[ty * head_dim_padded + d] = acc;
    }
    l[ty] = l_prev + l_new;
    m[ty] = m_new;
    __syncthreads();
  }

  const int out_offset = qkv_offset + q_tile_idx * head_dim;
  const bool valid_row = (q_tile_idx + ty) < seq_len;
  const float l_norm = l[ty];

#pragma unroll 4
  for (int d = 0; d < head_dim; d += blockDim.x) {
    const int d_idx = d + tx;
    if (d_idx < head_dim && valid_row) {
      const float val = o[ty * head_dim_padded + d_idx] / l_norm;
      O[out_offset + (ty * head_dim) + d_idx] = val;
    }
  }

  // Write L - which is the max of each row of query, this will be used in
  // backward pass
  if (tx == 0 && ty < Br && valid_row) {
    L[batch_head_idx * seq_len + q_tile_idx + ty] = l_norm;
    M[batch_head_idx * seq_len + q_tile_idx + ty] = m[ty];
  }
}

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
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
      int idx = qkv_offset + (q_tile_idx + i) * head_dim + d;
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
      for (int d = 0; d < head_dim; d++) {
        sum += O_tile[q * head_dim + d] * dO_tile[q * head_dim + d];
      }
      D[batch_head_idx * seq_len + q_tile_idx + q] = sum;
    }
  }
}

__global__ void compute_dK_dV(const float *Q, const float *K, float *V,
                              const float *dO, const float *M, const float *L,
                              const float *D, float *dK, float *dV,
                              float softmax_scale, int batch_size,
                              int num_heads, int seq_len, int head_dim, int Bc,
                              int Br, bool causal) {
  int kv_tile_idx = blockIdx.x * Bc;
  int batch_head_idx = blockIdx.y;
  int batch = batch_head_idx / num_heads;
  int head = batch_head_idx % num_heads;
  int qkv_offset =
      batch * num_heads * seq_len * head_dim + head * seq_len * head_dim;
  int head_dim_padded = head_dim + (head_dim % 32 == 0 ? 1 : 0);

  extern __shared__ float smem[];
  float *K_tile = (float *)smem;
  float *V_tile = K_tile + Bc * head_dim_padded;
  float *Q_tile = V_tile + Bc * head_dim_padded;
  float *S = Q_tile + Br * head_dim_padded;
  float *dS = S + Br * Bc;

  // load K, V tile
  for (int i = threadIdx.y; i < Bc && (kv_tile_idx + i) < seq_len;
       i += blockDim.y) {
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
      int idx = qkv_offset + (kv_tile_idx + i) * head_dim + d;
      K_tile[i * head_dim_padded + d] = K[idx];
      V_tile[i * head_dim_padded + d] = V[idx];
    }
  }
  __syncthreads();

  // Iterate over query tiles
  for (int j = 0; j < (seq_len + Br - 1) / Br; j++) {
    int q_tile_idx = j * Br;

    // load Q tile
    for (int i = threadIdx.y; i < Br && (q_tile_idx + i) < seq_len;
         i += blockDim.y) {
      for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        int idx = qkv_offset + (q_tile_idx + i) * head_dim + d;
        Q_tile[i * head_dim_padded + d] = Q[idx];
      }
    }
    __syncthreads();

    // Compute S = exp(softmax_scale * Q @ K^T - M)
    for (int q = threadIdx.y; q < Br && (q_tile_idx + q) < seq_len;
         q += blockDim.y) {
      int q_idx = q_tile_idx + q;
      float m_val = M[batch_head_idx * seq_len + q_idx];
      for (int k = threadIdx.x; k < Bc && (kv_tile_idx + k) < seq_len;
           k += blockDim.x) {
        if (causal && q_idx < (kv_tile_idx + k))
          continue;
        float qk = 0.0f;
        for (int d = 0; d < head_dim; d++) {
          qk +=
              Q_tile[q * head_dim_padded + d] * K_tile[k * head_dim_padded + d];
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
            dv += S[q * Bc + k] * dO[qkv_offset + q_idx * head_dim + d];
          }
        }
        atomicAdd(&dV[qkv_offset + k_idx * head_dim + d], dv);
      }
    }

    // Compute dS = P * (dO @ V^T - D)
    for (int q = threadIdx.y; q < Br && (q_tile_idx + q) < seq_len;
         q += blockDim.y) {
      int q_idx = q_tile_idx + q;
      float Di = D[batch_head_idx * seq_len + q_idx];
      for (int k = threadIdx.x; k < Bc && (kv_tile_idx + k) < seq_len;
           k += blockDim.x) {
        if (causal && q_idx < (kv_tile_idx + k))
          continue;
        int k_idx = kv_tile_idx + k;
        float dp = 0.0f;
        for (int d = 0; d < head_dim; d++) {
          dp += dO[qkv_offset + q_idx * head_dim_padded + d] *
                V_tile[k * head_dim_padded + d];
        }
        dS[q * Bc + k] = S[q * Bc + k] * (dp - Di);
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
            dk += dS[q * Bc + k] * Q_tile[q * head_dim_padded + d];
          }
        }
        atomicAdd(&dK[qkv_offset + k_idx * head_dim + d], dk * softmax_scale);
      }
    }
  }
}

__global__ void compute_dQ(const float *Q, const float *K, const float *V,
                           const float *dO, const int num_heads,
                           const int seq_len, const int head_dim,
                           const int Br) {
  const int q_tile_idx = blockIdx.x * Br;
  const int batch_head_idx = blockIdx.y;
  const int batch = batch_head_idx / num_heads;
  const int head = batch_head_idx % num_heads;
  const int qkv_offset =
      batch * num_heads * seq_len * head_dim + head * seq_len + head_dim;

  extern __shared__ float smem[];
  float *Q_tile = smem;
  float *dO_tile = Q_tile + Br * head_dim;
  float *dQ_tile = dO_tile + Br * head_dim;

  // Load Q, dO tiles and init dQ
  for (int i = threadIdx.y; i < Br && (q_tile_idx + i) < seq_len;
       i += blockDim.y) {
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
      int idx = qkv_offset + (qkv_offset + i) * head_dim + d;
      Q_tile[i * head_dim + d] = Q[idx];
      dO_tile[i * head_dim + d] = dO[idx];
      dQ_tile[i * head_dim + d] = 0.0f;
    }
  }
  __syncthreads();


}

// Host pybind functions
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
fa2_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, bool causal) {
  const int Br = 2;
  const int Bc = 2;

  const int batch_size = Q.size(0);
  const int num_heads = Q.size(1);
  const int seq_len = Q.size(2);
  const int head_dim = Q.size(3);

  const int head_dim_padded = head_dim + (head_dim % 32 == 0 ? 1 : 0);
  const int Bc_padded = Bc + (Bc % 32 == 0 ? 1 : 0);

  const int Tr = (seq_len + Br - 1) / Br;
  const int Tc = (seq_len + Bc - 1) / Bc;

  const float softmax_scale = 1 / sqrt(head_dim);

  auto O = torch::zeros_like(Q);
  auto M = torch::zeros({batch_size, num_heads, seq_len}, Q.options());
  auto L = torch::zeros({batch_size, num_heads, seq_len}, Q.options());
  auto S_debug =
      torch::empty({batch_size, num_heads, seq_len, seq_len}, Q.options());
  auto P_debug =
      torch::empty({batch_size, num_heads, seq_len, seq_len}, Q.options());

  torch::Device device(torch::kCUDA);
  O = O.to(device);
  M = M.to(device);
  L = L.to(device);

  dim3 grid_size(Tr, batch_size * num_heads, 1);
  dim3 block_size(Bc, Br, 1);

  size_t smem_size = ((2 * Br + 2 * Bc) * head_dim_padded) * sizeof(float) +
                     (2 * Br) * sizeof(float) + Bc_padded * sizeof(float);

  fa2_fwd<<<grid_size, block_size, smem_size>>>(
      Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
      O.data_ptr<float>(), M.data_ptr<float>(), L.data_ptr<float>(),
      S_debug.data_ptr<float>(), P_debug.data_ptr<float>(), softmax_scale,
      num_heads, batch_size, seq_len, head_dim, Br, Bc, Tr, Tc, causal);

  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  return std::make_tuple(O, M, L, S_debug, P_debug);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fa2_backward(torch::Tensor dO, torch::Tensor Q, torch::Tensor K,
             torch::Tensor V, torch::Tensor O, torch::Tensor M, torch::Tensor L,
             bool causal) {
  const int Br = 16;
  const int Bc = 16;

  const int batch_size = Q.size(0);
  const int num_heads = Q.size(1);
  const int seq_len = Q.size(2);
  const int head_dim = Q.size(3);

  const int head_dim_padded = head_dim + (head_dim % 32 == 0 ? 1 : 0);
  const int Bc_padded = Bc + (Bc % 32 == 0 ? 1 : 0);

  const int Tr = (seq_len + Br - 1) / Br;
  const int Tc = (seq_len + Bc - 1) / Bc;

  const float softmax_scale = 1.0f / sqrt(static_cast<float>(head_dim));

  auto dQ = torch::zeros_like(Q);
  auto dK = torch::zeros_like(K);
  auto dV = torch::zeros_like(V);

  float *D;

  CHECK_CUDA(cudaMalloc(&D, batch_size * num_heads * seq_len * sizeof(float)));

  dim3 grid_D((seq_len + 15) / 16, batch_size * num_heads);
  dim3 block_D(32, 16);
  size_t smem_D = 16 * head_dim * 2 * sizeof(float);
  compute_D<<<grid_D, block_D, smem_D>>>(O.data_ptr<float>(),
                                         dO.data_ptr<float>(), D, batch_size,
                                         num_heads, seq_len, head_dim);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  dim3 grid_dK_dV((seq_len + Bc - 1) / Bc, batch_size * num_heads);
  dim3 block_dK_dV(32, 16);
  size_t smem_size_dk_dv =
      (Bc * head_dim_padded * 2 + 2 * Br * head_dim_padded + 2 * Br * Bc) *
      sizeof(float);
  compute_dK_dV<<<grid_dK_dV, block_dK_dV, smem_size_dk_dv>>>(
      Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
      dO.data_ptr<float>(), M.data_ptr<float>(), L.data_ptr<float>(), D,
      dK.data_ptr<float>(), dV.data_ptr<float>(), softmax_scale, batch_size,
      num_heads, seq_len, head_dim, Bc, Br, causal);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  dim3 grid_dq((seq_len + Br - 1) / Br, batch_size * num_heads);
  dim3 block_dq(32, 16);
  size_t smem_size_dq =
      (4 * Br * head_dim_padded + 2 * Bc * head_dim_padded + 2 * Br * Bc) *
      sizeof(float);
  compute_dQ<<<grid_dq, block_dq, smem_size_dq>>>(
      Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
      dO.data_ptr<float>(), M.data_ptr<float>(), L.data_ptr<float>(), D,
      dQ.data_ptr<float>(), softmax_scale, batch_size, num_heads, seq_len,
      head_dim, Br, Bc, causal);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaFree(D));

  return std::make_tuple(dQ, dK, dV);
}
