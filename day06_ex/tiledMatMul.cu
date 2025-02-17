#include<stdio.h>
#define TILE_WIDTH 2

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


__global__ void tiledMatrixMultiplication(float *A, float *B, float *C, int W){
  __shared__ float Ads[TILE_WIDTH][TILE_WIDTH]; 
  __shared__ float Bds[TILE_WIDTH][TILE_WIDTH];

  int tx = threadIdx.x; int ty = threadIdx.y;
  int bx = blockIdx.x; int by = blockIdx.y;

  int Row = ty + TILE_WIDTH * by;
  int Col = tx + TILE_WIDTH * bx;

  float Pvalue = 0;
  for(int ph = 0; ph < (W/TILE_WIDTH); ph ++){
    Ads[ty][tx] = A[Row*W + ph*TILE_WIDTH + tx];
    Bds[ty][tx] = B[(ph*TILE_WIDTH + ty) * W + Col];

    __syncthreads();
    for (int k = 0; k< TILE_WIDTH; k++){
      Pvalue += Ads[ty][k] * Bds[k][tx];  
    }
    __syncthreads();
  }

  C[Row*W + Col] = Pvalue;
}


void tiledMatMul(float *A, float*B, float*C, int W){
  float *A_d, *B_d, *C_d;
  cudaMalloc(&A_d, W*W * sizeof(float));  
  cudaMalloc(&B_d, W*W * sizeof(float));
  cudaMalloc(&C_d, W*W * sizeof(float));

  cudaMemcpy(A_d, A, W*W * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B, W*W * sizeof(float), cudaMemcpyHostToDevice);
 
  dim3 blockDim(2, 2);
  dim3 gridDim(ceil(W/2), ceil(W/2));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  tiledMatrixMultiplication<<<gridDim, blockDim>>>(A_d, B_d, C_d, W);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float et;
  cudaEventElapsedTime(&et, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaMemcpy(C, C_d, W*W * sizeof(float), cudaMemcpyDeviceToHost);
  printf("GPU time for TILED MatMul = %f ms\n", et);

  printf("The result is: \n");
    for (int i=0; i<W; i++){
      for (int j=0; j<W; j++){
        printf("%.2f ", C[i * W + j]);
      } 
      printf("\n");
    }
  printf("\n");
}

void naiveMatMul(float *A, float*B, float*C, int W){
  float *A_d, *B_d, *C_d;
  cudaMalloc(&A_d, W*W * sizeof(float));  
  cudaMalloc(&B_d, W*W * sizeof(float));
  cudaMalloc(&C_d, W*W * sizeof(float));

  cudaMemcpy(A_d, A, W*W * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B, W*W * sizeof(float), cudaMemcpyHostToDevice);
 
  dim3 blockDim(2, 2);
  dim3 gridDim(ceil(W/2), ceil(W/2));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  matMulKernel<<<gridDim, blockDim>>>(A_d, B_d, C_d, W);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float et;
  cudaEventElapsedTime(&et, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaMemcpy(C, C_d, W*W * sizeof(float), cudaMemcpyDeviceToHost);
  printf("GPU time for NAIVE matMul = %f ms\n", et);

  printf("The result is: \n");
    for (int i=0; i<W; i++){
      for (int j=0; j<W; j++){
        printf("%.2f ", C[i * W + j]);
      } 
      printf("\n");
    }
    printf("\n");
}

int main(){
  int W = 4;
  float *A = (float *)malloc(W*W * sizeof(float));  
  float *B = (float *)malloc(W*W * sizeof(float));
  float *C = (float *)malloc(W*W * sizeof(float));

  for (int i=0; i<W; i++){
    for (int j=0; j<W; j++){
      A[i*W + j] = 1.0;
      B[i*W + j] = 1.0;
    }  
  }

  naiveMatMul(A, B, C, W);
  tiledMatMul(A, B, C, W);
     return 0;
}
