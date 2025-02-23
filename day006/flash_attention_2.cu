#include <cmath>
__global__ void fa2_forward(int num_heads, int batch_size, int Br, int Bc,
                            int head_dim) {
  int block_idx = blockIdx.x;
  int batch_head_idx = blockIdx.y;
  int batch_idx = ceil(batch_head_idx / num_heads);
  int head_idx = batch_head_idx % num_heads;

  int batch_offset = batch_idx * batch_size;
  int head_offset = head_idx * num_heads;
  int qkv_offset = batch_offset + head_offset;

  extern __shared__ float smem[];
  float *Q_tile = smem;
  float *K_tile = Q_tile + (Br * head_dim);
  float *V_tile = K_tile + (Bc * head_dim);
}

int main() {
  int batch_size = 1;
  int num_heads = 1;
  int seq_len = 3;
  int head_dim = 2;

  int Br = 2;
  int Bc = 2;

  int Tr = ceil((float)seq_len / Br);
  int Tc = ceil((float)seq_len / Bc);

  float softmax_scale = 1 / std::sqrt(head_dim);

  dim3 grid_size(Tr, batch_size * num_heads, 1);
  dim3 block_size(Bc, Br, 1);

  int smem = 2 * (Br * head_dim) + 2 * (Bc * head_dim);
  fa2_forward<<<grid_size, block_size, smem>>>(num_heads, batch_size, Br, Bc,
                                               head_dim);
}
