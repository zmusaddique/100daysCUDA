#define NUM_BINS 7
#define COARSE_FACTOR 2

__global__ void histogram_pvt_coarsed(float *data, int length, int *histo) {
  __shared__ int histo_s[NUM_BINS];
  for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
    histo_s[i] = 0u;
  }
  __syncthreads();

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = tid * COARSE_FACTOR; i < min((tid + 1) * COARSE_FACTOR, length);
       i++) { // coarse the factor stride; // This is not coalesced access
    // upper bound is the stride or the length of data
    int alpha_pos = data[i] - 'a';
    if (alpha_pos >= 0 && alpha_pos < 26) {
      atomicAdd(&(histo_s[alpha_pos / 4]), 1);
    }
  }
  __syncthreads();
  for (int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
    int binval = histo_s[bin];
    if (binval > 0) {
      atomicAdd(&(histo[bin]), binval);
    }
  }
}

__global__ void histogram_agg(char *data, int length, int *histo) {
  __shared__ int histo_s[NUM_BINS];
  for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
    histo_s[i] = 0u;
  }
  __syncthreads();

  int accumulator = 0;
  int prev_bin = -1;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = tid; i < length; i += blockDim.x * gridDim.x) {
    int alpha_pos = data[i] - 'a';
    if (alpha_pos >= 0 && alpha_pos < 26) {
      int bin = alpha_pos / 4;
      if (bin == prev_bin) {
        ++accumulator;
      } else {
        if (accumulator > 0) {
          atomicAdd(&(histo_s[prev_bin]), accumulator);
        }
        accumulator = 1;
        prev_bin = bin;
      }
    }
  }

  if (accumulator > 0) {
    atomicAdd(&histo_s[prev_bin], accumulator);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
    int binval = histo_s[i];
    if (binval > 0) {
      atomicAdd(&histo[i], binval);
    }
  }
}
