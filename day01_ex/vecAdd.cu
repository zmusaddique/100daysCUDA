#include<stdio.h>

__global__
void vecAddKernel(float *A_d, float* B_d, float* C_d, int n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (i<n){
    C_d[i]  = A_d[i] + B_d[i];
  }
}  

void vedAdd(float *A_h, float * B_h, float *C_h, int n){
  // Part 1: Allocate the mem in the device
  //         Copy from host mem to device mem
  int size = n * sizeof(float);
  float *A_d, *B_d, *C_d;

  cudaMalloc((void**)&A_d, size);
  cudaMalloc((void**)&B_d, size);
  cudaMalloc((void**)&C_d, size);

  cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

  // Part 2: Call kernel & compute 
  vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);

  cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

  //Part3: Free the memory in device
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);  
}

int main(){
  float A[] = {1.0, 2.0, 3.0};
  float B[] = {1.0, 2.0 , 3.0};
  float C[3];
  int n = 3;

  vedAdd(A, B, C, n);

  for(int i = 0; i<n; i++){
    printf("%f ", C[i])  ;
  }

  return 0;
}

