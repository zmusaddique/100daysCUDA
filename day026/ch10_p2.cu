#define BLOCK_DIM 32
#define COARSE_FACTOR 2

__global__ void convergent_reduction_smem(float *in, int length, float *out) {
  __shared__ float smem[BLOCK_DIM];
  int t = threadIdx.x;
  smem[t] = in[t] + in[t + BLOCK_DIM]; // improves latency

  for (int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
    __syncthreads(); // I feel this might bottleneck
    if (threadIdx.x < stride) {
      smem[t] += in[t + stride];
    }
  }
  if (threadIdx.x == 0) {
    *out = smem[0];
  }
}

__global__ void multiblock_reduction(float *in, int length, float *out) {
  __shared__ float smem[BLOCK_DIM];
  int t = threadIdx.x;
  int segment = 2 * blockIdx.x * blockDim.x;
  int i = segment + threadIdx.x;
  smem[t] = in[i] + in[i + BLOCK_DIM]; // loading the segment; rest's same

  for (int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
    __syncthreads();
    if (t < stride) {
      smem[t] += smem[t + stride];
    }
  }

  if (t == 0) {
    atomicAdd(out, smem[0]);
  }
}

__global__ void reduction_coarsed(float *in, int length, float *out) {
  __shared__ float smem[BLOCK_DIM];
  int t = threadIdx.x;
  int segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
  int i = segment + t;
  float sum = in[i];

  for (int tile = 1; tile < COARSE_FACTOR * 2; tile++) { // x2 coz two ele are loaded
    sum += in[i + tile * BLOCK_DIM];
  }
  smem[t] = sum;

  for (int stride = blockIdx.x / 2; stride >= 1; stride /= 2) {
    __syncthreads();
    if (t < stride) {
      smem[t] += in[t + stride];
    }
  }

  if (t == 0) {
    atomicAdd(out, smem[0]);
  }
}
