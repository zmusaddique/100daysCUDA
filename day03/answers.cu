#include<stdio.h>

// Chapter 3

// 1. 
// a. Write a kernel that has each thread produce one output matrix row. Fill in
//    the execution configuration parameters for the design 

// Intuition: Element of resultant matrix C[i,j] is the dot product of 2 vectors
// ie. i'th row of A and j'th column of B.  

__global__ 
void matMulRowKernel(float * A, float *B, float *C, int n) { /// n - for a square matrix
  // int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < n){
    for (int col= 0; col<n; col++){  
      int Cval = 0;

      for (int k = 0; k < n; k++){
        Cval += A[row*n + k] * B[k*n + col];
      }
      C[row*n + col] = Cval;
    }
  } 
}
// Explanation: It is same as matMul kernel. In this case, to produce an entire
// row of resultant matrix, we need to map a single thread to a single row. Then 
// iterate over all columns resultant matrix. This sol assumes inputs to be square 
// matrices for simplicity. 


// 1.
// b. Write a kernel that has each thread produce one output matrix column. Fill
//    in the execution configuration parameters for the design.

__global__ 
void matMulColKernel(float * A, float *B, float *C, int n) { /// n - for a square matrix
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  // int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < n){
    for (int row= 0; row<n; row++){  
      int Cval = 0;

      for (int k = 0; k < n; k++){
        Cval += A[row*n + k] * B[k*n + col];
      }
      C[row*n + col] = Cval;
    }
  } 
}

// Explanation: same as above but for column


// 1.
// c. Analyze the pros and cons of each of the two kernel designs

// Pros: Might be less overhead of mapping indices
// Cons: Parallelism is not utilized properly as a single thread is used for 
//       multiple elements of the output. Better map them with overhead. 


int main(){
  int N = 3;
  float *A, *B, *C;

  A = (float *)malloc(N*N * sizeof(float));
  B = (float *)malloc(N*N * sizeof(float));
  C = (float *)malloc(N*N * sizeof(float));

  for (int i = 0; i< N; i++){
    for (int j = 0; j<N; j++){
      A[i*N + j] = 1.0f;
      B[i*N + j] = 3.0f;
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

  // matMulRowKernel<<<gridDim, blockDim>>>( A_d, B_d, C_d, N); 
  matMulColKernel<<<gridDim, blockDim>>>( A_d, B_d, C_d, N); 

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
//=============================================================================
// 2. 
// 
__global__
void matVecKernel(float *B, float *C, float *A, int n){
  row = blockIdx.y * blockId.y + threadIdx.y;
  col = blockIdx.x * blockId.x + threadIdx.x;

  int Aval = 0;
  for(int j=0; j<n; j++){
    Aval += B[row*n + j] * C[j]; 
  } 

  A[row * n + col] = Aval;

}


// 3. 
// a. 16 * 32 = 512 threads per block
// b. (300-1)/16 + 1, (150-1)/32 + 1 = 19, 5 => THREADS = 19*5 * 512 = 48640
// c. 19 * 5 = 95
// d. 300 * 150 = 45000 threads


// 4.  width of 400 and a height of 500; element at row 20 and column 10:
// a. row major = row * width + col => 20 * 400 + 10 = 8010 
// b. column major = col * height + row => 10 * 500 + 20 = 5020


// 5. width of 400, a height of 500, and a depth of 300. element at x 5 10, y 20, and z 5
// W*H*z + W*y + x = 400*500*5 + 400*20 + 10 = 1008010

