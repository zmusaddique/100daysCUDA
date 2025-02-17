// Attention = softmax((Q*Kt)/sqrt(dk))*V
// We are interested in optimizing softmax, not matMULs
//

// Safe softmax = e^(Zi - k)/ sum(Ezi - k), k - max val

// steps in softmax -
// 1. Find max val - 1 iteration
// 2. Calculate normalization factor - 1 iteration
// 3. Apply softmax - 1 iteration

// All of these are sequential and require previous steps to be executed

// However by being stubborn to reduce the passes we arrive to just 2 passes
// by using  online softmax and max in on

// 1. m= -inf
//    l0 =

// for i = 1 to N
//    mi = max(mi-1, Xi)
//    li = li-1* e^(mi-1 - mi) + e^(Xi-mi)

// for k=1 to N
//  softmax with current values

// we need BATCH_size * HEADs * (SEQ_LEN/BlockSize)
#include <__clang_cuda_builtin_vars.h>
__global__ void flash_attention_forward(
  const float * Q, const float *K, const float *V, float *O, int N, int d, int Br, int Bc){
  
  int row_block_index = blockIdx.x; // The i in the outer loop
  int row_index_in_block = threadIdx.y; // The row in Br x d block
  int col_index_in_block = threadIdx.x; // The column in Br x d block

  if (row_index_in_block >= Br || col_index_in_block >= Bc) return;

  int global_row_index = row_index_in_block + row_block_index * Br;
  if (global_row_index > N) return;

  // Step 4: Load Qi from HBM to SRAM
  __shared__ float Qi_sram[Br*d];
  for (int j_local=0; j_local < d; j_local += blockIdx.x){
    int global_col_index = j_local + threadIdx.x;
    if (global_col_index<d){
      Qi_sram[global_row_index*d + global_col_index] = Q[global_row_index * d + global_col_index];
    }
  }
  __syncthreads(); // Make sure the row-block is loaded entirely

  // Step5: intialize O, l and m 
  float O_i_current_val = 0.0f;
  float l_i_current_val = 0.0f;
  float m_i_current_val = 0.0f;

  // Steps 6-11
  for (int col_block_index=0; col_block_index < Tc; ++col_block_index){
    __shared__ float Kj_sram[Bc * d];
    __shared__ float Vj_sram[Bc * d];
    // Load K & V

    __syncthreads();

    float S_ij = 0.0f;
    for (int k=0; k<d; k++){
      S_ij += Qi_sram[row_index_in_block*d+k] * K[];
    }

    // Mj, Pi, li
    // Oi
    __syncthreads();
    

    // Apply normalization factor to Oi
    // correct Li
  }
// Back to HBM


} 
