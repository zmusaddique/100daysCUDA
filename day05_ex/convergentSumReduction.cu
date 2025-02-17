__global__ void sumReductionKernel(float *a, float *b){
  int i = 2*threadIdx.x;
  for (int stride =1; stride<Bi.x; stride *= 2){
    if (threadIdx.x % stride == 0){
      a[i] += a[i+stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0){
    *b = a[0]; 
  }
}
