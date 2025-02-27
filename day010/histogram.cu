#include <cuda_runtime.h>
#include <stdio.h>

#define NUM_BINS 256 // Histogram bins
#define BLOCK_SIZE 256

__global__ void histogram_kernel(const unsigned char *d_input, int *d_histogram,
                                 int data_size) {
  __shared__ int local_hist[NUM_BINS]; // Shared memory histogram

  // Initialize shared memory
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int local_tid = threadIdx.x;

  if (local_tid < NUM_BINS)
    local_hist[local_tid] = 0;
  __syncthreads();

    if (tid < data_size) {
    atomicAdd(&local_hist[d_input[tid]], 1);
  }
  __syncthreads();


  if (local_tid < NUM_BINS) {
    atomicAdd(&d_histogram[local_tid], local_hist[local_tid]);
  }
}

int main() {
  int data_size = 1024 * 1024; // 1MB image-like data
  unsigned char *h_input = (unsigned char *)malloc(data_size);
  int *h_histogram = (int *)calloc(NUM_BINS, sizeof(int));


  for (int i = 0; i < data_size; i++) {
    h_input[i] = rand() % NUM_BINS;
  }

  unsigned char *d_input;
  int *d_histogram;
  cudaMalloc(&d_input, data_size);
  cudaMalloc(&d_histogram, NUM_BINS * sizeof(int));
  cudaMemset(d_histogram, 0, NUM_BINS * sizeof(int));

  cudaMemcpy(d_input, h_input, data_size, cudaMemcpyHostToDevice);

  int grid_size = (data_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  histogram_kernel<<<grid_size, BLOCK_SIZE>>>(d_input, d_histogram, data_size);

  cudaMemcpy(h_histogram, d_histogram, NUM_BINS * sizeof(int),
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < 10; i++) {
    printf("Bin %d: %d\n", i, h_histogram[i]);
  }

  cudaFree(d_input);
  cudaFree(d_histogram);
  free(h_input);
  free(h_histogram);

  return 0;
}
