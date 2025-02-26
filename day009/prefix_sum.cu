#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SECTION_SIZE 1024
#define MAX_BLOCKS 65535

// Error checking macro
#define cudaCheckError()                                                       \
  {                                                                            \
    cudaError_t e = cudaGetLastError();                                        \
    if (e != cudaSuccess) {                                                    \
      printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,                     \
             cudaGetErrorString(e));                                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

__global__ void Kogge_Stone_scan_kernel(int *X, int *Y, unsigned int N) {
  __shared__ int XY[SECTION_SIZE];
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N) {
    XY[threadIdx.x] = X[i];
  } else {
    XY[threadIdx.x] = 0;
  }

  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
    __syncthreads();
    int temp;
    if (threadIdx.x >= stride)
      temp = XY[threadIdx.x] + XY[threadIdx.x - stride];
    __syncthreads();
    if (threadIdx.x >= stride)
      XY[threadIdx.x] = temp;
  }

  if (i < N) {
    Y[i] = XY[threadIdx.x];
  }
}

__global__ void combine_block_results(int *Y, unsigned int N,
                                      unsigned int blockSize) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (blockIdx.x > 0 && i < N) {
    unsigned int blockOffset = blockIdx.x * blockSize;
    if (blockOffset < N) {
      int sum = Y[blockOffset - 1];
      Y[i] += sum;
    }
  }
}

void cpu_inclusive_scan(int *input, int *output, unsigned int N) {
  output[0] = input[0];
  for (unsigned int i = 1; i < N; i++) {
    output[i] = output[i - 1] + input[i];
  }
}

void init_random_data(int *data, unsigned int N) {
  for (unsigned int i = 0; i < N; i++) {
    data[i] = rand() % 100; // Generates integers between 0 and 99
  }
}

bool verify_results(int *cpu_result, int *gpu_result, unsigned int N) {
  for (unsigned int i = 0; i < N; i++) {
    if (cpu_result[i] != gpu_result[i]) {
      printf("Mismatch at index %u: CPU = %d, GPU = %d\n", i, cpu_result[i],
             gpu_result[i]);
      return false;
    }
  }
  return true;
}

void run_gpu_scan(int *h_input, int *h_output, unsigned int N) {
  int *d_input, *d_output;

  cudaMalloc((void **)&d_input, N * sizeof(int));
  cudaMalloc((void **)&d_output, N * sizeof(int));
  cudaCheckError();

  cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaCheckError();

  dim3 blockSize(SECTION_SIZE);
  dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

  if (gridSize.x > MAX_BLOCKS) {
    printf("Error: Grid size exceeds maximum allowed blocks\n");
    cudaFree(d_input);
    cudaFree(d_output);
    return;
  }

  Kogge_Stone_scan_kernel<<<gridSize, blockSize>>>(d_input, d_output, N);
  cudaCheckError();

  if (gridSize.x > 1) {
    combine_block_results<<<gridSize, blockSize>>>(d_output, N, blockSize.x);
    cudaCheckError();
  }

  cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);
  cudaCheckError();

  cudaFree(d_input);
  cudaFree(d_output);
}

int main(int argc, char **argv) {
  unsigned int N = 100;

  if (argc > 1) {
    N = atoi(argv[1]);
  }

  printf("Performing inclusive scan on array of size %u\n", N);

  int *h_input = (int *)malloc(N * sizeof(int));
  int *h_output_gpu = (int *)malloc(N * sizeof(int));
  int *h_output_cpu = (int *)malloc(N * sizeof(int));

  srand(time(NULL));
  init_random_data(h_input, N);

  clock_t cpu_start = clock();
  cpu_inclusive_scan(h_input, h_output_cpu, N);
  clock_t cpu_end = clock();
  double cpu_time =
      ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0; // ms

  clock_t gpu_start = clock();
  run_gpu_scan(h_input, h_output_gpu, N);
  clock_t gpu_end = clock();
  double gpu_time =
      ((double)(gpu_end - gpu_start)) / CLOCKS_PER_SEC * 1000.0; // ms

  bool results_match = verify_results(h_output_cpu, h_output_gpu, N);

  printf("CPU time: %.2f ms\n", cpu_time);
  printf("GPU time: %.2f ms\n", gpu_time);
  printf("Speedup: %.2fx\n", cpu_time / gpu_time);
  printf("Results %s\n", results_match ? "MATCH" : "DO NOT MATCH");

  printf("\nFirst 10 elements of the scan:\n");
  printf("Index\tInput\tCPU Output\tGPU Output\n");
  for (int i = 0; i < 10 && i < N; i++) {
    printf("%d\t%d\t%d\t\t%d\n", i, h_input[i], h_output_cpu[i],
           h_output_gpu[i]);
  }

  free(h_input);
  free(h_output_gpu);
  free(h_output_cpu);

  return 0;
}
