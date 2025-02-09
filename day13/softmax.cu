#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat> // Include this header for FLT_MAX

// Warp-level max reduction
template <typename T>
__device__ T warpReduceMax(T val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        T tmp = __shfl_down_sync(0xffffffff, val, offset);
        val = max(val, tmp);
    }
    return val;
}

// Warp-level sum reduction
template <typename T>
__device__ T warpReduceSum(T val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-wide max reduction
template <typename T>
__device__ T blockMaxReduce(T val) {
    __shared__ T shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    
    val = warpReduceMax(val);
    
    if (lane == 0)
        shared[wid] = val;
    
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : -FLT_MAX;
    if (wid == 0)
        val = warpReduceMax(val);
    
    return val;
}

// Block-wide sum reduction
template <typename T>
__device__ T blockSumReduce(T val) {
    __shared__ T shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    
    val = warpReduceSum(val);
    
    if (lane == 0)
        shared[wid] = val;
    
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
    if (wid == 0)
        val = warpReduceSum(val);
    
    return val;
}

__global__ void softmax_kernel(float* input, float* output, int N) {
    extern __shared__ float shared_mem[];
    float* exponents = shared_mem;
    
    int row_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    float* row_input = input + row_idx * N;
    float* row_output = output + row_idx * N;

    // Step 1: Find max value in the row
    float thread_max = -FLT_MAX;
    for (int i = tid; i < N; i += blockDim.x) {
        thread_max = max(thread_max, row_input[i]);
    }
    
    float row_max = blockMaxReduce(thread_max);
    
    // Step 2: Compute exponentials and sum
    float thread_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        exponents[i] = expf(row_input[i] - row_max);
        thread_sum += exponents[i];
    }
    
    float row_sum = blockSumReduce(thread_sum);
    
    // Step 3: Normalize and write output
    for (int i = tid; i < N; i += blockDim.x) {
        row_output[i] = exponents[i] / row_sum;
    }
}

void verify_softmax(float* input, float* output, int batch_size, int feature_size) {
    for (int b = 0; b < batch_size; b++) {
        float* row = input + b * feature_size;
        float* out_row = output + b * feature_size;
        
        // Compute CPU softmax
        float max_val = -INFINITY;
        for (int i = 0; i < feature_size; i++) {
            max_val = fmaxf(max_val, row[i]);
        }
        
        float sum = 0.0f;
        float* exp_values = new float[feature_size];
        for (int i = 0; i < feature_size; i++) {
            exp_values[i] = expf(row[i] - max_val);
            sum += exp_values[i];
        }
        
        // Compare with GPU results
        float tolerance = 1e-5;
        for (int i = 0; i < feature_size; i++) {
            float cpu_val = exp_values[i] / sum;
            float gpu_val = out_row[i];
            
            if (fabs(cpu_val - gpu_val) > tolerance) {
                printf("Mismatch at batch %d, element %d: CPU=%.6f, GPU=%.6f\n",
                       b, i, cpu_val, gpu_val);
                exit(1);
            }
        }
        delete[] exp_values;
    }
    printf("All results verified successfully!\n");
}

int main() {
    const int batch_size = 1024;
    const int feature_size = 256;
    const size_t bytes = batch_size * feature_size * sizeof(float);
    
    // Allocate host memory
    float* h_input = new float[batch_size * feature_size];
    float* h_output = new float[batch_size * feature_size];
    
    // Initialize input with random values
    for (int i = 0; i < batch_size * feature_size; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f - 5.0f;
    }
    
    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    
    // Copy data to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int block_size = 256;
    dim3 grid(batch_size);
    dim3 block(block_size);
    size_t shared_mem_size = feature_size * sizeof(float);
    
    // Check device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (shared_mem_size > prop.sharedMemPerBlock) {
        printf("Required shared memory exceeds device capability!\n");
        exit(1);
    }
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    softmax_kernel<<<grid, block, shared_mem_size>>>(d_input, d_output, feature_size);
    cudaEventRecord(stop);
    
    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    
    // Copy results back
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    
    // Print timing results
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %.3f ms\n", milliseconds);
    
    // Verify results
    verify_softmax(h_input, h_output, batch_size, feature_size);
    
    // Cleanup
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}
