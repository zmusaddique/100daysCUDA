#define NUM_BINS 7

__global__ void histo_pvt_kernel(float *data, int length, int *histo) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < length) {
    int alpha_pos = data[i] - 'a';
    if (alpha_pos >= 0 && alpha_pos < 26) {
      atomicAdd(&(histo[blockIdx.x * NUM_BINS + alpha_pos / 4]), 1);
    }
  }
  if (blockIdx.x > 0) {
    __syncthreads();
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
      unsigned int binValue = histo[blockIdx.x * NUM_BINS + bin];
      if (binValue > 0) {
        atomicAdd(&(histo[bin]), binValue);
      }
    }
  }
}

__global__ void histo_pvt_shared(char *data, int length, unsigned int *histo) {
  __shared__ unsigned int histo_s[NUM_BINS];
  for (int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
    histo_s[bin] = 0u;
  }
  __syncthreads();
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < length) {
    int alpha_pos = data[i] - 'a';
    if (alpha_pos >= 0 && alpha_pos < 26) {
      atomicAdd(&histo_s[alpha_pos / 4], 1);
    }
  }

  __syncthreads();
  for (int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
    int binval = histo_s[bin];
    if (binval > 0) {
      atomicAdd(&histo[bin], binval);
    }
  }
}
