#define SECTION_SIZE 1024 // is also = N

__global__ void brent_kung(float *in, float *out, int N) {

  for (int stride = 1; stride <= blockDim.x; stride *= 2) {
    __syncthreads();
    if ((threadIdx.x + 1) % (2 * stride) == 0) { // This will produce a massive control divergence, real massive
      in[threadIdx.x] += in[threadIdx.x - stride];
    }
  }
}

__global__ void brent_kung_less_divergence(float *in, float *out, int N){

  for (int stride = 1; stride < blockDim.x; stride *=2){
    __syncthreads();
    int i = (threadIdx.x + 1) * (2 * stride) - 1;
    if (i < SECTION_SIZE){ // Now entire warps will be shutdown gradually
      in[i] += in[i - stride];
    }
  }

  // reverse-tree
  for (int stride = SECTION_SIZE / 4; stride >0; stride /=2){
    __syncthreads();
    int i = (threadIdx.x + 1) * (2 * stride) -1;
    if (stride + i < SECTION_SIZE){
      in[i + stride] += in[i];
    }
  } 
}


// Complete brent-kung
__global__ void brent_kung_complete(float *in, float *out, int  N){
  __shared__ float X[SECTION_SIZE];
  int i = 2 * blockIdx.x * blockDim.x + threadIdx.x; // We do this becoz we cover 2 elements in one thread
  if (i < N) X[threadIdx.x] = in[i];
  if (i + blockDim.x < N) X[threadIdx.x + blockDim.x] = in[i + blockDim.x];
  
  for (int stride = 1; stride <= blockDim.x; stride *= 2){
    __syncthreads();
    int index = (threadIdx.x + 1) * 2 * stride -1;
    if (index < N) X[index] += X[index - stride];
  }

  for (int stride = 1; stride <= blockDim.x; stride *= 2){
    __syncthreads();
    int index = (threadIdx.x + 1) * 2 * stride -1;
    if (index < N) X[index + stride] += X[index];
  }
  __syncthreads();
  if (i < N) out[i] = X[threadIdx.x];
  if (i + blockDim.x < N) out[i + blockDim.x] = X[threadIdx.x + blockDim.x];
}
