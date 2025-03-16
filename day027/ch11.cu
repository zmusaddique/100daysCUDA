#define SECTION_SIZE 32

__global__ void kogge_stone(float *x, float *y, int N) {
  __shared__ float XY[SECTION_SIZE];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    XY[threadIdx.x] = x[i];
  } else {
    XY[threadIdx.x] = 0.0f;
  }
  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    __syncthreads(); // all elements must be synchronized before beginning an
                     // iteration
    float temp;
    if (threadIdx.x >= stride) {
      temp = XY[threadIdx.x] + XY[threadIdx.x - stride];
    }
    __syncthreads(); // To prevent write-after-read; bottlenecks too
    if (threadIdx.x >= stride) {
      XY[threadIdx.x] = temp;
    }
  }
  if (i < N) {
    y[i] = XY[threadIdx.x];
  }
}

__global__ void kogge_stone_double_buffer(float *x, float *y, int N) {
  __shared__ float buffer1[SECTION_SIZE];
  __shared__ float buffer2[SECTION_SIZE];// remove WAR, writer-after-read race 

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  buffer1[threadIdx.x] = i < N ? x[i] : 0.0f;
  __syncthreads();
  
  float *input_s = buffer1;
  float *output_s = buffer2;

  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    if (threadIdx.x >= stride) {
      output_s[threadIdx.x] =
          input_s[threadIdx.x] + input_s[threadIdx.x - stride];
    } else {
      output_s[threadIdx.x] = input_s[threadIdx.x];
    }
    float *tmp = input_s;
    input_s = output_s;
    output_s = tmp;
  }
  if (i < N) {
    y[i] = input_s[threadIdx.x];
  }
}
