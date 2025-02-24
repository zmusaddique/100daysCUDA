#include <cmath>
#include <stdio.h>
#include <torch/extension.h>
#include <tuple>

__global__ void fa2_fwd(float *Q, float *K, float *V, float *O, float *M,
                        float softmax_scale, int num_heads, int batch_size,
                        int seq_len, int head_dim, int Br, int Bc, int Tr,
                        int Tc, bool causal) {
  int ty = threadIdx.y;
  int tx = threadIdx.x;

  int block_idx_q = blockIdx.x; // tile ID
  int batch_head_idx = blockIdx.y;
  int batch_idx = batch_head_idx / num_heads;
  int head_idx = batch_head_idx % num_heads;

  int q_tile_idx = blockIdx.x * Br;

  // i * stride
  int batch_offset = batch_idx * (num_heads * seq_len * head_dim);
  int head_offset = head_idx * (seq_len * head_dim);
  // qkv_offset is used to skip the sequences and head dimensions from the
  // starting pointed loc
  int qkv_offset = batch_offset + head_offset;

  extern __shared__ float smem[];
  float *Q_tile = (float *)smem;
  float *K_tile = Q_tile + (Br * head_dim);
  float *V_tile = K_tile + (Bc * head_dim);
  float *l = V_tile + (Bc * head_dim);
  float *m = l + Br;
  float *o = m + Br;
  float *qk = o + (Br * head_dim);

  for (int e = tx; e < Bc; e += blockDim.x) {
    qk[e] = 0.0f;
  }
  __syncthreads();

  // Load Q tile
  for (int i = ty; i < Br; i += blockDim.y) {
    int row = q_tile_idx + i;
    for (int j = tx; j < head_dim; j += blockDim.x) {
      if (row < seq_len) {
        Q_tile[i * head_dim + j] = Q[qkv_offset + row * head_dim + j];
      } else {
        Q_tile[i * head_dim + j] = 0.0f;
      }
    }
  }

  // Accumulators
  if (threadIdx.x == 0 && threadIdx.y < Br) {
    m[threadIdx.y] = -INFINITY;
    l[threadIdx.y] = 0.0f;
  }
  for (int c = tx; c < head_dim; c += blockDim.x) {
    o[threadIdx.y * head_dim + c] = 0.0f;
  }
  __syncthreads();

  // Looping over KV_tiles
  for (int j = 0; j < Tc; j++) {
    int j_start = j * Bc;
    // Load K, V
    for (int i = ty; i < Bc; i += blockDim.y) {
      const int row = j_start + i;
      for (int c = tx; c < head_dim; c += blockDim.x) {
        if (row < seq_len) {
          K_tile[i * head_dim + c] = K[qkv_offset + row * head_dim + c];
          V_tile[i * head_dim + c] = V[qkv_offset + row * head_dim + c];
        } else {
          K_tile[i * head_dim + c] = 0.0f;
          V_tile[i * head_dim + c] = 0.0f;
        }
      }
    }

    // Resetting qk for each new kv tile
    for (int k = 0; k < Bc; k++) {
      qk[threadIdx.y * Bc + k] = 0.0f;
    }

    __syncthreads();

    // QK^T
    for (int d = 0; d < head_dim; d++) {
      const float qval = Q_tile[threadIdx.y * head_dim + d];
      for (int k = 0; k < Bc; k++) {
        qk[threadIdx.y * Bc + k] += qval * K_tile[k * head_dim + d];
      }
    }

    // Apply softmax
    float max_val = -INFINITY;
    for (int k = 0; k < Bc; k++) {
      const int global_k = j_start + k;
      float val = qk[threadIdx.y * Bc + k];
      if (causal && (q_tile_idx + threadIdx.y) < global_k) {
        val = -INFINITY;
      } else {
        val *= softmax_scale;
        max_val = fmaxf(max_val, qk[threadIdx.y * Bc + k]);
      }
      qk[threadIdx.y * Bc + k] = val;
    }

    float m_prev = m[threadIdx.y];
    float m_new = fmaxf(m_prev, max_val);
    float exp_scale = expf(m_prev - m_new);

    float l_prev = l[threadIdx.y] * expf(m_prev - m_new);
    float l_new = 0.0f;
    for (int k = 0; k < Bc; k++) {
      l_new += expf(qk[threadIdx.y * Bc + k] - m_new);
    }

    const float alpha = expf(m_prev - m_new);
    for (int d = 0; d < head_dim; d++) {
      float acc = o[threadIdx.y * head_dim + d] * alpha;
      for (int k = 0; k < Bc; k++) {
        float p = expf(qk[threadIdx.y * Bc + k] - m_new);
        acc += p * V_tile[k * head_dim + d];
      }
      o[threadIdx.y * head_dim + d] = acc;
    }
    l[threadIdx.y] = l_prev + l_new;
    m[threadIdx.y] = m_new;
    __syncthreads();
  }

  int out_offset = qkv_offset + q_tile_idx * head_dim;
  for (int d = 0; d < head_dim; d += blockDim.x) {
    if ((d + threadIdx.x) < head_dim) {
      const float l_norm = l[threadIdx.y];
      const float val = o[threadIdx.y * head_dim + (threadIdx.x + d)] / l_norm;
      if ((q_tile_idx + threadIdx.y) < seq_len) {
        O[out_offset + (threadIdx.y * head_dim) + d + threadIdx.x] = val;
      }
    }
  }

  if (threadIdx.x == 0 && threadIdx.y < Br) {
    M[batch_head_idx * seq_len + q_tile_idx + threadIdx.y] =
        m[threadIdx.y] + logf(l[threadIdx.y]);
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
                     (2 * Br) * sizeof(float) + Bc * sizeof(float);
  // fa2_fwd<<<grid_size, block_size, smem_size>>>(num_heads, batch_size, Br,
  //                                                  Bc, head_dim);
}
