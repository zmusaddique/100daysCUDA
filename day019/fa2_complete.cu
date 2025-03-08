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
  const int batch = batch_head_idx / num_heads;
  const int head = batch_head_idx % num_heads;

  const int q_tile_idx = blockIdx.x * Br;

  const int batch_offset = batch * (num_heads * seq_len * head_dim);
  const int head_offset = head * (seq_len * head_dim);

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
  float *S = o + (Br * head_dim_padded);

  for (int i = ty; i < Br; i += blockDim.y) {
    for (int e = tx; e < Bc_padded; e += blockDim.x) {
      if (e < Bc)
        S[e] = 0.0f;
    }
  }
  __syncthreads();

// Load Q tile
#pragma unroll 4
  for (int i = ty; i < Br; i += blockDim.y) {
    int row = q_tile_idx + i;
    const bool valid_row = row < seq_len;

#pragma unroll 4
    for (int j = tx; j < head_dim; j += blockDim.x) {
      Q_tile[i * head_dim_padded + j] =
          valid_row ? Q[qkv_offset + row * head_dim + j] : 0.0f;
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
      S[row_offset + k] = 0.0f;
    }

    __syncthreads();

// QK^T
#pragma unroll 4
    for (int d = 0; d < head_dim; d++) {
      for (int k = 0; k < Bc; k++) {
        S[row_offset + k] +=
            Q_tile[ty * head_dim_padded + d] * K_tile[k * head_dim_padded + d];
      }
    }

    // Apply causal mask and scale in one pass
    float max_val = -INFINITY;
    const int q_global_idx = q_tile_idx + ty;

#pragma unroll 4
    for (int k = 0; k < Bc; k++) {
      const int k_global_idx = j_start + k;
      float val = S[row_offset + k];

      if (causal && q_global_idx < k_global_idx) {
        S[row_offset + k] = -INFINITY;
      } else {
        val *= softmax_scale;
        max_val = fmaxf(max_val, val);
        S[row_offset + k] = val;
      }
    }

    float m_prev = m[ty];
    float m_curr = fmaxf(m_prev, max_val);

    float l_prev_correction = __expf(m_prev - m_curr);
    float l_prev = l[ty] * l_prev_correction;
    float P_tilde = 0.0f;

#pragma unroll 4
    for (int k = 0; k < Bc; k++) {
      P_tilde += __expf(S[row_offset + k] - m_curr);
    }

#pragma unroll 4
    for (int d = 0; d < head_dim; d++) {
      float O_prev = o[ty * head_dim_padded + d] * l_prev_correction;

#pragma unroll 4
      for (int k = 0; k < Bc; k++) {
        const float p = __expf(S[row_offset + k] - m_curr);
        O_prev += p * V_tile[k * head_dim_padded + d];
      }
      o[ty * head_dim_padded + d] = O_prev;
    }
    l[ty] = l_prev + P_tilde;
    m[ty] = m_curr;
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
    M[batch_head_idx * seq_len + q_tile_idx + threadIdx.y] =
        m[threadIdx.y] + logf(l[threadIdx.y]);
  }
}

__global__ void compute_D(const float *O, const float *dO, float *D,
                          const int num_heads, const int seq_len,
                          const int head_dim, const int Br) {
  int q_tile_idx = blockIdx.x * Br;
  int batch_head_idx = blockIdx.y;
  int batch = batch_head_idx / num_heads;
  int head = batch_head_idx % num_heads;
  int qkv_offset =
      batch * num_heads * seq_len * head_dim + head * seq_len * head_dim;

  extern __shared__ float smem[];
  float *O_tile = (float *)smem;
  float *dO_tile = O_tile + Br * head_dim;

#pragma unroll
  for (int i = threadIdx.y; i < Br && (q_tile_idx + i) < seq_len;
       i += blockDim.y) {
#pragma unroll
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
      int idx = qkv_offset + (q_tile_idx + i) * head_dim + d;
      O_tile[i * head_dim + d] = O[idx];
      dO_tile[i * head_dim + d] = dO[idx];
    }
  }
  __syncthreads();

  // compute D
  if (threadIdx.x == 0) {
#pragma unroll
    for (int q = threadIdx.y; q < Br && (q_tile_idx + q) < seq_len;
         q += blockDim.y) {
      float sum = 0.0f;
#pragma unroll
      for (int d = 0; d < head_dim; d++) {
        sum += O_tile[q * head_dim + d] * dO_tile[q * head_dim + d];
      }
      D[batch_head_idx * seq_len + q_tile_idx + q] = sum;
    }
  }
}

__global__ void compute_dK_dV(const float *Q, const float *K, const float *V,
                              const float *dO, float *dK, float *dV,
                              const float *M, const float *D, float *P_out,
                              float *dS_out, const int num_heads,
                              const int seq_len, const int head_dim,
                              const float softmax_scale, const int Br,
                              const int Bc, const bool causal) {
  const int k_tile_idx = blockIdx.x * Bc;
  const int batch_head_idx = blockIdx.y;
  const int batch = batch_head_idx / num_heads;
  const int head = batch_head_idx % num_heads;

  const int qkv_offset =
      batch * num_heads * seq_len * head_dim + head * seq_len * head_dim;

  extern __shared__ float smem[];
  float *K_tile = smem;
  float *V_tile = K_tile + Bc * head_dim;
  float *Q_tile = V_tile + Bc * head_dim;
  float *P = Q_tile + Br * head_dim;
  float *dO_tile = P + Br * Bc;
  float *dS_tile = dO_tile + Br * head_dim;
  float *dK_tile = dS_tile + Br * Bc;
  float *dV_tile = dK_tile + Bc * head_dim;

  for (int i = threadIdx.y; i < Bc && (k_tile_idx + i) < seq_len;
       i += blockDim.y) {
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
      const int idx = qkv_offset + (k_tile_idx + i) * head_dim + d;
      K_tile[i * head_dim + d] = K[idx];
      V_tile[i * head_dim + d] = V[idx];
      dK_tile[i * head_dim + d] = 0.0f;
      dV_tile[i * head_dim + d] = 0.0f;
    }
  }
  __syncthreads();

  // Loop over Q tiles
  for (int i = 0; i < (seq_len + Br - 1) / Br; i++) {
    const int q_tile_idx = i * Br;

    for (int q = threadIdx.y; q < Br && (q_tile_idx + q) < seq_len;
         q += blockDim.y) {
      for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        Q_tile[q * head_dim + d] =
            Q[qkv_offset + (q_tile_idx + q) * head_dim + d];
        dO_tile[q * head_dim + d] =
            dO[qkv_offset + (q_tile_idx + q) * head_dim + d];
      }
    }
    __syncthreads();

    // P = exp(QK^T * softmax_scale - M)
    for (int q = threadIdx.y; q < Br && (q_tile_idx + q) < seq_len;
         q += blockDim.y) {
      const float m_val = M[batch_head_idx * seq_len + (q_tile_idx + q)];
      for (int k = threadIdx.x; k < Bc && (k_tile_idx + k) < seq_len;
           k += blockDim.x) {
        if (causal && (q_tile_idx + q) < (k_tile_idx + k)) {
          P[q * Bc + k] = 0.0f;
        } else {
          float qk = 0.0f;
          for (int d = 0; d < head_dim; d++) {
            qk += Q_tile[q * head_dim + d] * K_tile[k * head_dim + d];
          }
          qk = softmax_scale * qk - m_val;
          P[q * Bc + k] = __expf(qk);
        }
        // Write P to output array
        if (P_out != nullptr) {
          int global_q = q_tile_idx + q;
          int global_k = k_tile_idx + k;
          if (global_q < seq_len && global_k < seq_len) {
            P_out[batch_head_idx * seq_len * seq_len + global_q * seq_len +
                  global_k] = P[q * Bc + k];
          }
        }
      }
    }
    __syncthreads();

    // Reset dS_tile
    for (int q = threadIdx.y; q < Br && (q_tile_idx + q) < seq_len;
         q += blockDim.y) {
      for (int k = threadIdx.x; k < Bc && (k_tile_idx + k) < seq_len;
           k += blockDim.x) {
        dS_tile[q * Bc + k] = 0.0f;
      }
    }
    __syncthreads();

    // dS
    for (int q = threadIdx.y; q < Br && (q_tile_idx + q) < seq_len;
         q += blockDim.y) {
      const int global_q = q_tile_idx + q;
      const float Di = D[batch_head_idx * seq_len + global_q];

      for (int k = threadIdx.x; k < Bc && (k_tile_idx + k) < seq_len;
           k += blockDim.x) {
        const int global_k = k_tile_idx + k;

        if (causal && global_q < global_k) {
          dS_tile[q * Bc + k] = 0.0f;
        } else {
          float dp = 0.0f;
          for (int d = 0; d < head_dim; d++) {
            dp += dO_tile[q * head_dim + d] * V_tile[k * head_dim + d];
          }
          dS_tile[q * Bc + k] = P[q * Bc + k] * (dp - Di);
        }
        if (dS_out != nullptr && global_q < seq_len && global_k < seq_len) {
          dS_out[batch_head_idx * seq_len * seq_len + global_q * seq_len +
                 global_k] = dS_tile[q * Bc + k];
        }
      }
    }
    __syncthreads();

    // dV
    for (int k = threadIdx.x; k < Bc && (k_tile_idx + k) < seq_len;
         k += blockDim.x) {
      for (int d = 0; d < head_dim; d++) {
        float dv = 0.0f;
        for (int q = 0; q < Br && (q_tile_idx + q) < seq_len; q++) {
          if (!causal || (q_tile_idx + q) >= (k_tile_idx + k))
            dv += P[q * Bc + k] * dO_tile[q * head_dim + d];
        }
        dV_tile[k * head_dim + d] += dv;
      }
    }

    // dK
    for (int k = threadIdx.x; k < Bc && (k_tile_idx + k) < seq_len;
         k += blockDim.x) {
      for (int d = threadIdx.y; d < head_dim; d += blockDim.y) {
        float dk = 0.0f;
        for (int q = 0; q < Br && (q_tile_idx + q) < seq_len; q++) {
          if (!causal || (q_tile_idx + q) >= (k_tile_idx + k))
            dk += dS_tile[q * Bc + k] * Q_tile[q * head_dim + d];
        }
        dK_tile[k * head_dim + d] += dk * softmax_scale;
      }
    }
    __syncthreads();
  }
  for (int k = threadIdx.y; k < Bc && (k_tile_idx + k) < seq_len;
       k += blockDim.y) {
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
      atomicAdd(&dK[qkv_offset + (k_tile_idx + k) * head_dim + d],
                dK_tile[k * head_dim + d]);
      atomicAdd(&dV[qkv_offset + (k_tile_idx + k) * head_dim + d],
                dV_tile[k * head_dim + d]);
    }
  }
}

__global__ void compute_dQ(const float *Q, const float *K, const float *V,
                           const float *dO, const float *M, const float *D,
                           float *dQ, const float softmax_scale,
                           const int num_heads, const int seq_len,
                           const int head_dim, const int Br, const int Bc,
                           const bool causal) {
  const int q_tile_idx = blockIdx.x * Br;
  const int batch_head_idx = blockIdx.y;
  const int batch = batch_head_idx / num_heads;
  const int head = batch_head_idx % num_heads;
  const int qkv_offset =
      batch * num_heads * seq_len * head_dim + head * seq_len * head_dim;

  extern __shared__ float smem[];
  float *Q_tile = smem;
  float *dO_tile = Q_tile + Br * head_dim;
  float *dQ_tile = dO_tile + Br * head_dim;
  float *K_tile = dQ_tile + Br * head_dim;
  float *V_tile = K_tile + Bc * head_dim;
  float *P = V_tile + Bc * head_dim;
  float *dS_tile = P + Br * Bc;

  // Load Q, dO tiles and init dQ
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

  for (int j = 0; j < (seq_len + Bc - 1) / Bc; j++) {
    const int k_tile_idx = j * Bc;

    for (int i = threadIdx.y; i < Bc && (k_tile_idx + i) < seq_len;
         i += blockDim.y) {
      for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const int idx = qkv_offset + (k_tile_idx + i) * head_dim + d;
        K_tile[i * head_dim + d] = K[idx];
        V_tile[i * head_dim + d] = V[idx];
      }
    }
    __syncthreads();

    // Missing this was the culprit
    for (int q = threadIdx.y; q < Br && (q_tile_idx + q) < seq_len;
         q += blockDim.y) {
      for (int k = threadIdx.x; k < Bc && (k_tile_idx + k) < seq_len;
           k += blockDim.x) {
        dS_tile[q * Bc + k] = 0.0f;
      }
    }
    __syncthreads();

    // P = exp(softmax * Q @ K^T - M)
    for (int q = threadIdx.y; q < Br && (q_tile_idx + q) < seq_len;
         q += blockDim.y) {
      float m_val = M[batch_head_idx * seq_len + q_tile_idx + q];
      for (int k = threadIdx.x; k < Bc && (k_tile_idx + k) < seq_len;
           k += blockDim.x) {
        if (causal && (q_tile_idx + q) < (k_tile_idx + k)) {
          P[q * Bc + k] = 0.0f;
        } else {
          float qk = 0.0f;
          for (int d = 0; d < head_dim; d++) {
            qk += Q_tile[q * head_dim + d] * K_tile[k * head_dim + d];
          }
          qk = softmax_scale * qk - m_val;
          P[q * Bc + k] = __expf(qk);
        }
      }
    }
    __syncthreads();

    for (int q = threadIdx.y; q < Br && (q_tile_idx + q) < seq_len;
         q += blockDim.y) {
      const int q_idx = q_tile_idx + q;
      const float Di = D[batch_head_idx * seq_len + q_idx];
      for (int k = threadIdx.x; k < Bc && (k_tile_idx + k) < seq_len;
           k += blockDim.x) {
        if (causal && q_idx < (k_tile_idx + k)) {
          dS_tile[q * Bc + k] = 0.0f;
        } else {
          float dp = 0.0f;
          for (int d = 0; d < head_dim; d++) {
            dp += dO_tile[q * head_dim + d] * V_tile[k * head_dim + d];
          }
          dS_tile[q * Bc + k] = P[q * Bc + k] * (dp - Di);
        }
      }
    }
    __syncthreads();

    for (int q = threadIdx.y; q < Br && (q_tile_idx + q) < seq_len;
         q += blockDim.y) {
      for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float dq = 0.0f;
        for (int k = 0; k < Bc && (k_tile_idx + k) < seq_len; k++) {
          if (causal && (q_tile_idx + q) < (k_tile_idx + k))
            continue;
          dq += dS_tile[q * Bc + k] * K_tile[k * head_dim + d];
        }
        dQ_tile[q * head_dim + d] += softmax_scale * dq;
      }
    }
    __syncthreads();
  }

  for (int i = threadIdx.y; i < Br && (q_tile_idx + i) < seq_len;
       i += blockDim.y) {
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
      const int idx = qkv_offset + (q_tile_idx + i) * head_dim + d;
      atomicAdd(&dQ[idx], dQ_tile[i * head_dim + d]);
    }
  }
}

// Host pybind functions
std::tuple<torch::Tensor, torch::Tensor>
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

  return std::make_tuple(O, M);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
fa2_backward(torch::Tensor dO, torch::Tensor Q, torch::Tensor K,
             torch::Tensor V, torch::Tensor O, torch::Tensor M, bool causal) {
  const int Br = 4;
  const int Bc = 4;

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

  auto D = torch::zeros(
      {batch_size, num_heads, seq_len},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

  auto P_out = torch::zeros(
      {batch_size, num_heads, seq_len, seq_len},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

  auto dS_out = torch::zeros(
      {batch_size, num_heads, seq_len, seq_len},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

  dim3 grid_D((seq_len + Br - 1) / Br, batch_size * num_heads);
  dim3 block_D(32, 16);
  size_t smem_D = Br * head_dim * 2 * sizeof(float);
  compute_D<<<grid_D, block_D, smem_D>>>(
      O.data_ptr<float>(), dO.data_ptr<float>(), D.data_ptr<float>(), num_heads,
      seq_len, head_dim, Br);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  dim3 grid_dK_dV((seq_len + Bc - 1) / Bc, batch_size * num_heads);
  dim3 block_dK_dV(16, 16);
  size_t smem_size_dk_dv =
      (4 * Bc * head_dim + 2 * Br * head_dim + 2 * Br * Bc) * sizeof(float);
  compute_dK_dV<<<grid_dK_dV, block_dK_dV, smem_size_dk_dv>>>(
      Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
      dO.data_ptr<float>(), dK.data_ptr<float>(), dV.data_ptr<float>(),
      M.data_ptr<float>(), D.data_ptr<float>(), P_out.data_ptr<float>(),
      dS_out.data_ptr<float>(), num_heads, seq_len, head_dim, softmax_scale, Br,
      Bc, causal);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  dim3 grid_dq((seq_len + Br - 1) / Br, batch_size * num_heads);
  dim3 block_dq(32, 16);
  size_t smem_size_dq =
      (3 * Br * head_dim + 2 * Bc * head_dim + 2 * Br * Bc) * sizeof(float);
  compute_dQ<<<grid_dq, block_dq, smem_size_dq>>>(
      Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
      dO.data_ptr<float>(), M.data_ptr<float>(), D.data_ptr<float>(),
      dQ.data_ptr<float>(), softmax_scale, num_heads, seq_len, head_dim, Br, Bc,
      causal);

  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  // CHECK_CUDA(cudaFree(D));

  return std::make_tuple(dQ, dK, P_out, dS_out, dV);
}

// todo: try making M tiled
