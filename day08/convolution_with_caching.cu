#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define FILTER_RADIUS 2
#define IN_TILE_DIM 32
#define OUT_TILE_DIM (IN_TILE_DIM - (2 * FILTER_RADIUS))

__constant__ float F[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];

// CUDA kernel for tiled convolution
__global__ void convolution_with_tiling(float *A, float *R, int width, int height) {
    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;

    __shared__ float A_s[IN_TILE_DIM][IN_TILE_DIM];

    if (row >= 0 && row < height && col >= 0 && col < width) {
        A_s[threadIdx.y][threadIdx.x] = A[row * width + col];
    } else {
        A_s[threadIdx.y][threadIdx.x] = 0.0f;
    }

    int tileRow = threadIdx.y - FILTER_RADIUS;
    int tileCol = threadIdx.x - FILTER_RADIUS;

    __syncthreads();

    if (row >= 0 && row < height && col >= 0 && col < width) {
        if (tileRow >= 0 && tileRow < OUT_TILE_DIM && tileCol >= 0 && tileCol < OUT_TILE_DIM) {
            float Rval = 0.0f;

            for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++) {
                for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {
                    Rval += F[fRow][fCol] * A_s[tileRow + fRow][tileCol + fCol];
                }
            }
            R[row * width + col] = Rval;
        }
    }
}

// CPU reference implementation for verification
void cpu_convolution(float *A, float *R, float *F, int width, int height, int r) {
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

// CUDA error checking macro
#define CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

int main() {
    int width = 512;
    int height = 512;

    size_t size = width * height * sizeof(float);

    // Allocate host memory
    float *A = (float *)malloc(size);
    float *R_gpu = (float *)malloc(size);
    float *R_cpu = (float *)malloc(size);
    float *F_h = (float *)malloc((2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1) * sizeof(float));

    // Initialize input matrix with checkerboard pattern
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            A[i * width + j] = (i + j) % 2 == 0 ? 1.0f : 0.0f;
        }
    }

    // Initialize filter with checkerboard pattern
    for (int i = 0; i < 2 * FILTER_RADIUS + 1; i++) {
        for (int j = 0; j < 2 * FILTER_RADIUS + 1; j++) {
            F_h[i * (2 * FILTER_RADIUS + 1) + j] = (i + j) % 2 == 0 ? 1.0f : 0.0f;
        }
    }

    // Allocate device memory
    float *A_d, *R_d;
    CHECK(cudaMalloc((void **)&A_d, size));
    CHECK(cudaMalloc((void **)&R_d, size));

    // Copy data to device
    CHECK(cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(F, F_h, (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1) * sizeof(float)));

    // Set up block and grid dimensions
    dim3 blockDims(IN_TILE_DIM, IN_TILE_DIM);
    dim3 gridDims((width + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

    // Launch kernel
    convolution_with_tiling<<<gridDims, blockDims>>>(A_d, R_d, width, height);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK(cudaMemcpy(R_gpu, R_d, size, cudaMemcpyDeviceToHost));

    // Run CPU reference implementation
    cpu_convolution(A, R_cpu, F_h, width, height, FILTER_RADIUS);

    // Verify GPU result against CPU result
    float max_error = 0.0f;
    for (int i = 0; i < width * height; i++) {
        float diff = fabs(R_gpu[i] - R_cpu[i]);
        if (diff > max_error) max_error = diff;
    }
    printf("Maximum error between GPU and CPU: %f\n", max_error);

    // Print a small portion of the result for visual inspection
    printf("GPU result (first 8x8 block):\n");
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            printf("%.2f ", R_gpu[i * width + j]);
        }
        printf("\n");
    }

    // Free memory
    free(A);
    free(R_gpu);
    free(R_cpu);
    free(F_h);
    CHECK(cudaFree(A_d));
    CHECK(cudaFree(R_d));

    return 0;
}
