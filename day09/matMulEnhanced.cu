#include <stdio.h>

#define TILE_SIZE 2

__global__ void matmul_generalized(float *A, float *B, float *R, int width) {
 
  extern __shared__ float A_B_s[];

  int tx = threadIdx.x; int ty = threadIdx.y;
  int bx = blockIdx.x; int by = blockIdx.y;

  int Row = by * blockDim.y + ty;
  int Col = bx * blockDim.x + tx;

  float Rval = 0.0f;

  for (int ph = 0; ph < (width + TILE_SIZE - 1)/TILE_SIZE; ph++){

    int ACol = ph * TILE_SIZE + tx;
    if (Row < width && ACol < width)
      A_B_s[ty*TILE_SIZE + tx] = A[Row * width + ACol];
    else 
      A_B_s[ty*TILE_SIZE + tx] = 0.0f;

    int BRow = ph  * TILE_SIZE + ty;
    if (Col < width && BRow < width)
      A_B_s[TILE_SIZE*TILE_SIZE + ty * TILE_SIZE + tx]= B[BRow * width + Col];
    else
      A_B_s[TILE_SIZE*TILE_SIZE + ty * TILE_SIZE + tx] = 0.0f; 
    __syncthreads(); // read-after-write dependency 

    for (int i = 0; i<TILE_SIZE; i++){
      Rval+=A_B_s[ty*TILE_SIZE + i] * A_B_s[TILE_SIZE*TILE_SIZE + i * TILE_SIZE + tx];    
    }
    __syncthreads(); // write-after-read dependency
  }
 
  if (Row<width && Col < width)     
    R[Row*width + Col] = Rval;  
}

int main(){
 int N = 3;
  float *A, *B, *R;
  
  int size = N*N * sizeof(float);

  A = (float *)malloc(size);
  B = (float *)malloc(size);
  R = (float *)malloc(size);

  for (int i = 0; i< N; i++){
    for (int j = 0; j<N; j++){
      A[i*N + j] = 1.0f;
      B[i*N + j] = 2.0f;
      R[i*N + j] = 0.0f;
    }
  }

  float *A_d, *B_d, *R_d;
 
  cudaMalloc((void **)&A_d, size);
  cudaMalloc((void **)&B_d, size);
  cudaMalloc((void **)&R_d, size);

  cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice); 
  cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice); 
  cudaMemcpy(R_d, R, size, cudaMemcpyHostToDevice); 

  dim3 blockDim(TILE_SIZE, TILE_SIZE);
  dim3 gridDim((N+blockDim.x-1)/blockDim.x, (N+blockDim.y-1)/blockDim.y);

  size_t sharedMemSize = 2 * TILE_SIZE * TILE_SIZE * sizeof(float); 
  matmul_generalized<<<gridDim, blockDim, sharedMemSize>>>(A_d, B_d, R_d, N); 

  cudaMemcpy(R, R_d, size, cudaMemcpyDeviceToHost);


  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(R_d);

  // Result
  printf("The resultant matrix R is: \n");
  for (int i = 0; i< N; i++){
    for (int j=0; j < N; j++){
      printf("%.2f ", R[i*N +j]);
    }
    printf("\n");
  }

  printf("The first matrix A was: \n");
  for (int i = 0; i< N; i++){
    for (int j=0; j < N; j++){
      printf("%.2f ", A[i*N +j]);
    }
    printf("\n");
  }

  printf("The second matrix B was: \n");
  for (int i = 0; i< N; i++){
    for (int j=0; j < N; j++){
      printf("%.2f ", B[i*N +j]);
    }
    printf("\n");
  }

  free(A);
  free(B);
  free(R);
}
