
void fa2_bwd(const float *Q, const float *K, float *V, const float *O,
             const float *M, const float *dO, float *dQ, float *dK, float *dV,
             float softmax_scale, int batch_size, int num_heads, int seq_len,
             int head_dim, int Br, int Bc, bool causal) {
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
  size_t smem_size_dK_dV = (Bc * head_dim_padded * 2 + Br * head_dim) * sizeof(float);
  compute_dK_dV<<<grid_dK_dV, block_dK_dV, smem_size_dK_dV>>>(
      Q, K, V, dO, M, D, dK, dV, softmax_scale, batch_size, num_heads, seq_len,
      head_dim, Bc, causal);
  CHECK_CUDA(cudaDeviceSynchronize());

  dim3 grid_dQ((seq_len + Br - 1) / Br, batch_size * num_heads);
  dim3 block_dQ(32, 16);
  size_t smem_size_dq =
      (Br * head_dim * 3 + Bc * head_dim * 2 + 2 * Br * Bc) * sizeof(float);
  compute_dQ<<<grid_dQ, block_dQ, smem_size_dq>>>(
      Q, K, V, dO, M, D, dQ, softmax_scale, batch_size, num_heads, seq_len,
      head_dim, Br, Bc, causal);
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaFree(D);
}
