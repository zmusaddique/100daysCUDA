#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_DIM 16 

#define CHECK_CUDA(call) \
do { \
  cudaError_t err = call; \
  if (err != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    exit(EXIT_FAILURE); \
  } \
} while (0)


__global__ void tile_matrix_transpose(float *input, float * output, int width, int height) {
  __shared__ float tile[BLOCK_DIM][BLOCK_DIM + 1];

  int input_x = blockIdx.x * BLOCK_DIM + threadIdx.x;
  int input_y = blockIdx.y * BLOCK_DIM + threadIdx.y;
  int input_idx = input_y * width + input_x;

  int output_x = blockIdx.y * BLOCK_DIM + threadIdx.x;
  int output_y = blockIdx.x * BLOCK_DIM + threadIdx.y;
  int output_idx = output_y * height + output_x;

  if (input_x < width && input_y < height) {
    tile[threadIdx.y][threadIdx.x] = input[input_idx];
  }

  __syncthreads();
  
  if (output_x < height && output_y < width){
    output[output_idx] = tile[threadIdx.x][threadIdx.y]; // transposed index 
  }
}


int main() {
    int width = 1024; 
    int height = 1024;
 
    size_t size = width * height * sizeof(float);
    float *input_h = (float *)malloc(size);
    float *output_h = (float *)malloc(size);
    float *expected_h = (float *)malloc(size);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            input_h[y * width + x] = y * width + x; 
        }
    }

    float *input_d, *output_d;
    CHECK_CUDA(cudaMalloc((void **)&input_d, size));
    CHECK_CUDA(cudaMalloc((void **)&output_d, size));

    CHECK_CUDA(cudaMemcpy(input_d, input_h, size, cudaMemcpyHostToDevice));

    dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
    dim3 gridDim((width + BLOCK_DIM - 1) / BLOCK_DIM, (height + BLOCK_DIM - 1) / BLOCK_DIM);

    tile_matrix_transpose<<<gridDim, blockDim>>>(input_d, output_d, width, height);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMemcpy(output_h, output_d, size, cudaMemcpyDeviceToHost));

    bool correct = true;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            expected_h[x * height + y] = input_h[y * width + x]; // Expected transposed value
            if (output_h[x * height + y] != expected_h[x * height + y]) {
                correct = false;
                break;
            }
        }
        if (!correct) break;
    }

    if (correct) {
        std::cout << "Transpose is correct!" << std::endl;
    } else {
        std::cout << "Transpose is incorrect!" << std::endl;
    }

    CHECK_CUDA(cudaFree(input_d));
    CHECK_CUDA(cudaFree(output_d));

    free(input_h);
    free(output_h);
    free(expected_h);

    return 0;
}
