#include <stdio.h>

#define TILE_DIM 32 
#define FILTER_RADIUS 2
#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2 * (FILTER_RADIUS))

__constant__ float F_c[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];

__global__ void tiled_convolution(float *A, float *R, int width, int height) {
  int col = threadIdx.x + OUT_TILE_DIM * blockIdx.x -
            FILTER_RADIUS; // These are indices for input element
  int row = threadIdx.y + OUT_TILE_DIM * blockIdx.y -
            FILTER_RADIUS; // We minus the natural radius offset

  __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];

  if (row >= 0 && row < height && col >= 0 && col < width) {
    N_s[threadIdx.y][threadIdx.x] = A[row * width + col];
  } else {
    N_s[threadIdx.y][threadIdx.x] = 0.0f;
  }

  __syncthreads(); // All aboard??

  int tileRow = threadIdx.y - FILTER_RADIUS; // Mapping threads to output tile
  int tileCol = threadIdx.x - FILTER_RADIUS;

  if (row >= 0 && row < height && col >= 0 &&
      col < width) { // filter from input data
    if (tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >= 0 &&
        tileRow < OUT_TILE_DIM) { // filter threads out of output tile mapping
      float Rval = 0.0f;
      for (int fRow = 0; fRow < (2 * FILTER_RADIUS + 1); fRow++) {
        for (int fCol = 0; fCol < (2 * FILTER_RADIUS + 1); fCol++) {
          Rval += F_c[fRow][fCol] * N_s[tileRow + fRow][tileCol + fCol];
        }
      }
      int out_col = blockIdx.x * OUT_TILE_DIM + tileCol;
      int out_row = blockIdx.y * OUT_TILE_DIM + tileRow;
      if (out_col >= 0 && out_col < width && out_row >= 0 && out_row < height) {
        R[out_row * width + out_col] = Rval;
      }
    }
  }
}

__global__ void convolution_cached_tiled(float *A, float *R, int width,
                                         int height) {
  int row = threadIdx.y + blockIdx.y * TILE_DIM;
  int col = threadIdx.x + blockIdx.x * TILE_DIM;

  __shared__ float A_s[TILE_DIM][TILE_DIM];

  if (row < height && col < width) {
    A_s[threadIdx.y][threadIdx.x] = A[row * width + col];
  } else {
    A_s[threadIdx.y][threadIdx.x] = 0.0f;
  }

  __syncthreads();

  if (row < height && col < width) {
    float Rval = 0.0f;
    for (int fRow = 0; fRow < (2 * FILTER_RADIUS + 1); fRow++) {
      for (int fCol = 0; fCol < (2 * FILTER_RADIUS + 1); fCol++) {
        int shared_row = (int)threadIdx.y - FILTER_RADIUS + fRow;
        int shared_col = (int)threadIdx.x - FILTER_RADIUS + fCol;
        if (shared_col >= 0 && shared_col < TILE_DIM && shared_row >= 0 &&
            shared_row < TILE_DIM) { // halo cells
          Rval += F_c[fRow][fCol] * A_s[shared_row][shared_col];
        } else {
          int global_col = (int)col - FILTER_RADIUS + fCol;
          int global_row = (int)row - FILTER_RADIUS + fRow;

          if (global_col >= 0 && global_col < width && global_row >= 0 &&
              global_row < height) {
            Rval += F_c[fRow][fCol] * A[global_row * width + global_col];
          }
        }
      }
    }
    R[row * width + col] = Rval;
  }
}

void cpu_convolution(float *A, float *R, float *F, int width, int height,
                     int r) {
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      float val = 0.0f;
      for (int fy = 0; fy < 2 * r + 1; fy++) {
        for (int fx = 0; fx < 2 * r + 1; fx++) {
          int in_y = y - r + fy;
          int in_x = x - r + fx;
          if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
            val += F[fy * (2 * r + 1) + fx] * A[in_y * width + in_x];
          }
        }
      }
      R[y * width + x] = val;
    }
  }
}

#define CHECK(call)                                                            \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__,       \
             __LINE__);                                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

int main() {
  int width = 1024;
  int height = 1024;

  size_t size = width * height * sizeof(float);

  float *A = (float *)malloc(size);
  float *R_gpu_tiled = (float *)malloc(size);
  float *R_gpu_cached = (float *)malloc(size);
  float *R_cpu = (float *)malloc(size);
  float *F_h = (float *)malloc((2 * FILTER_RADIUS + 1) *
                               (2 * FILTER_RADIUS + 1) * sizeof(float));

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      A[i * width + j] = (i + j) % 2 == 0 ? 1.0f : 0.0f;
    }
  }

  for (int i = 0; i < 2 * FILTER_RADIUS + 1; i++) {
    for (int j = 0; j < 2 * FILTER_RADIUS + 1; j++) {
      F_h[i * (2 * FILTER_RADIUS + 1) + j] = (i + j) % 2 == 0 ? 1.0f : 0.0f;
    }
  }

  float *A_d, *R_d_tiled, *R_d_cached;
  CHECK(cudaMalloc((void **)&A_d, size));
  CHECK(cudaMalloc((void **)&R_d_tiled, size));
  CHECK(cudaMalloc((void **)&R_d_cached, size));

  CHECK(cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpyToSymbol(F_c, F_h,
                           (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1) *
                               sizeof(float)));

  dim3 blockDims(IN_TILE_DIM, IN_TILE_DIM);
  dim3 gridDims((width + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

  dim3 blockDims1(TILE_DIM, TILE_DIM);
  dim3 gridDims1((width + TILE_DIM - 1) / TILE_DIM,
                 (height + TILE_DIM - 1) / TILE_DIM);

  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));

  CHECK(cudaEventRecord(start));

  tiled_convolution<<<gridDims, blockDims>>>(A_d, R_d_tiled, width, height);

  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  CHECK(cudaEventRecord(stop));
  CHECK(cudaEventSynchronize(stop));

  float tiled_time = 0.0f;
  CHECK(cudaEventElapsedTime(&tiled_time, start, stop));
  printf("tiled_convolution execution time: %.3f ms\n", tiled_time);

  // Measure convolution_cached_tiled execution time
  CHECK(cudaEventRecord(start));

  convolution_cached_tiled<<<gridDims1, blockDims1>>>(A_d, R_d_cached, width,
                                                      height);

  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  CHECK(cudaEventRecord(stop));
  CHECK(cudaEventSynchronize(stop));

  float cached_time = 0.0f;
  CHECK(cudaEventElapsedTime(&cached_time, start, stop));
  printf("convolution_cached_tiled execution time: %.3f ms\n", cached_time);

  if (tiled_time < cached_time) {
    printf("tiled_convolution is faster by %.3f ms\n",
           cached_time - tiled_time);
  } else {
    printf("convolution_cached_tiled is faster by %.3f ms\n",
           tiled_time - cached_time);
  }

  CHECK(cudaMemcpy(R_gpu_tiled, R_d_tiled, size, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(R_gpu_cached, R_d_cached, size, cudaMemcpyDeviceToHost));

  cpu_convolution(A, R_cpu, F_h, width, height, FILTER_RADIUS);

  float max_error_tiled = 0.0f;
  float max_error_cached = 0.0f;
  for (int i = 0; i < width * height; i++) {
    float diff_tiled = fabs(R_gpu_tiled[i] - R_cpu[i]);
    float diff_cached = fabs(R_gpu_cached[i] - R_cpu[i]);
    if (diff_tiled > max_error_tiled)
      max_error_tiled = diff_tiled;
    if (diff_cached > max_error_cached)
      max_error_cached = diff_cached;
  }
  printf("Maximum error (tiled_convolution): %f\n", max_error_tiled);
  printf("Maximum error (convolution_cached_tiled): %f\n", max_error_cached);

  // Cleanup
  free(A);
  free(R_gpu_tiled);
  free(R_gpu_cached);
  free(R_cpu);
  free(F_h);
  CHECK(cudaFree(A_d));
  CHECK(cudaFree(R_d_tiled));
  CHECK(cudaFree(R_d_cached));
  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(stop));

  return 0;
}
