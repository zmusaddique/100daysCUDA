#include<stdio.h>

__global__ 
void matMulKernel(float *A , float * B, float * C, int width){ // only width because we assume a square matrix
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < width && col < width){
    float Cval = 0;
    for (int k = 0; k<width; k++){
      Cval += A[row*width+k] * B[k*width + col]; 
    }
    C[row*width+col] = Cval;
  }
}

int main(){
  int N = 3;
  float *A, *B, *C;

  A = (float *)malloc(N*N * sizeof(float));
  B = (float *)malloc(N*N * sizeof(float));
  C = (float *)malloc(N*N * sizeof(float));

  for (int i = 0; i< N; i++){
    for (int j = 0; j<N; j++){
      A[i*N + j] = 1.0f;
      B[i*N + j] = 2.0f;
      C[i*N + j] = 0.0f;
    }
  }

  float *A_d, *B_d, *C_d;
 
  int size = N*N * sizeof(float);
  cudaMalloc((void **)&A_d, size);
  cudaMalloc((void **)&B_d, size);
  cudaMalloc((void **)&C_d, size);

  cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice); 
  cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice); 
  cudaMemcpy(C_d, C, size, cudaMemcpyHostToDevice); 

  dim3 blockDim(16,16);
  dim3 gridDim((N+blockDim.x-1)/blockDim.x, (N+blockDim.y-1)/blockDim.y);

  matMulKernel<<<gridDim, blockDim>>>( A_d, B_d, C_d, N); 

  cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);


  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);

  // Result
  printf("The resultant matrix C is: \n");
  for (int i = 0; i< N; i++){
    for (int j=0; j < N; j++){
      printf("%.2f ", C[i*N +j]);
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
  free(C);
}
