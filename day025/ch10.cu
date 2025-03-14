__global__ void simple_reduction(float *in, float *output) {
  int i = 2 * threadIdx.x;
  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    if (threadIdx.x % stride == 0) {
      in[i] += in[i + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    *output = in[0];
  }
}

__global__ void simple_red_reimagined(float *in, float *out) {
  int i = threadIdx.x;
  for (int stride = blockDim.x; stride >= 1; stride /= 2) { // makes huge differences
    if (threadIdx.x < stride) {
      in[i] += in[i + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    *out = in[0];
  }
}
