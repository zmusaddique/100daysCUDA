#include <cmath>
#include <stdio.h>
#include <torch/extension.h>
#include <tuple>

__global__ void fa2_fwd(float *Q, float *K, float *V, float *O, float *M,
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
      O[out_offset + (threadIdx.y * head_dim) + d_idx] = val;
    }
  }

  if (tx == 0 && ty < Br && valid_row) {
    M[batch_head_idx * seq_len + q_tile_idx + ty] = m[ty] + __logf(l[ty]);
  }
}

std::tuple<torch::Tensor, torch::Tensor>
fa2_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
  const int Br = 2;
  const int Bc = 2;

  const int batch_size = Q.size(0);
  const int num_heads = Q.size(1);
  const int seq_len = Q.size(2);
  const int head_dim = Q.size(3);

  const int Tr = (seq_len + Br - 1) / Br;
  const int Tc = (seq_len + Bc - 1) / Bc;

  const float softmax_scale = 1 / sqrt(head_dim);

  const bool causal = false;

  auto O = torch::zeros_like(Q);
  auto M = torch::zeros({batch_size, num_heads, seq_len}, Q.options());

  torch::Device device(torch::kCUDA);
  O = O.to(device);
  M = M.to(device);

  dim3 grid_size(Tr, batch_size * num_heads, 1);
  dim3 block_size(Bc, Br, 1);

  size_t smem_size = ((2 * Br + 2 * Bc) * head_dim) * sizeof(float) +
                     (2 * Br) * sizeof(float) + Bc * sizeof(float);

  fa2_fwd<<<grid_size, block_size, smem_size>>>(
      Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
      O.data_ptr<float>(), M.data_ptr<float>(), softmax_scale, num_heads,
      batch_size, seq_len, head_dim, Br, Bc, Tr, Tc, causal);

  return std::make_tuple(O, M);
}

int main() {
  int batch_size = 1;
  int num_heads = 1;
  int seq_len = 3;
  int head_dim = 2;

  const int Br = 2;
  const int Bc = 2;

  const int Tr = ceil((float)seq_len / Br);
  const int Tc = ceil((float)seq_len / Bc);

  float softmax_scale = 1 / std::sqrt(head_dim);

  float Q_size = batch_size * num_heads * seq_len * head_dim;
  float K_size = batch_size * num_heads * seq_len * head_dim;
  float V_size = batch_size * num_heads * seq_len * head_dim;

  float *Q_h = (float *)malloc(Q_size * sizeof(float));

  dim3 grid_size(Tr, batch_size * num_heads, 1);
  dim3 block_size(Bc, Br, 1);

  size_t smem_size = ((2 * Br + 2 * Bc) * head_dim) * sizeof(float) +
                     (2 * Br) * sizeof(float) + (Bc * Br) * sizeof(float);
  // fa2_fwd<<<grid_size, block_size, smem_size>>>(num_heads, batch_size, Br,
  //                                                  Bc, head_dim);
}
