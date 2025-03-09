#include <stdio.h>

__global__ void matMulKernel(int *A, int *B, int *C, const int width,
                             int const tile_size) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int row = by * tile_size + ty;
  const int col = bx * tile_size + tx;

  extern __shared__ int smem[];
  int *As = smem;
  int *Bs = smem + tile_size * tile_size;
  int Cval = 0.0f;

  for (int ph = 0; ph < (width + tile_size - 1) / tile_size; ph++) {
    const int a_col = ph * tile_size + tx;
    const int b_row = ph * tile_size + ty;

    As[ty * tile_size + tx] =
        (row < width && a_col < width) ? A[row * width + a_col] : 0;
    Bs[tx * tile_size + ty] =
        (b_row < width && col < width) ? B[b_row * width + col] : 0;
    __syncthreads();

    for (int k = 0; k < tile_size; k++) {
      Cval += As[ty * tile_size + k] * Bs[k * tile_size + tx];
    }
    __syncthreads();
  }

  C[row * width + col] = Cval;
}

int main() {
  int width = 4;
  int tile_size = 2;
  int *A, *B, *C;
  int arr_size = width * width * sizeof(int);
  A = (int *)malloc(arr_size);
  B = (int *)malloc(arr_size);
  C = (int *)malloc(arr_size);

  for (int i = 0; i < width; i++) {
    for (int j = 0; j < width; j++) {
      A[i * width + j] = i * width + j + 1;
      B[j * width + i] = i * width + j + 1; // col major order
    }
  }

  int *A_d, *B_d, *C_d;
  cudaMalloc((void **)&A_d, arr_size);
  cudaMalloc((void **)&B_d, arr_size);
  cudaMalloc((void **)&C_d, arr_size);
  cudaMemcpy(A_d, A, arr_size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B, arr_size, cudaMemcpyHostToDevice);
  cudaMemset(C_d, 0, arr_size);

  dim3 blockSize(tile_size, tile_size, 1);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                (width + blockSize.y - 1) / blockSize.y, 1);
  size_t smem_size = 2 * tile_size * tile_size * sizeof(int);
  matMulKernel<<<gridSize, blockSize, smem_size>>>(A_d, B_d, C_d, width,
                                                   tile_size);
  cudaMemcpy(C, C_d, arr_size, cudaMemcpyDeviceToHost);
  printf("C \n");
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < width; j++) {
      printf("%d ", C[i * width + j]);
    }
    printf("\n");
  }
  return 0;
}
