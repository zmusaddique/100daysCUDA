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

int main(){
  float *a = malloc(16 * sizeof(float));
  float *b = malloc(sizeof(float));

  for (int i = 0; i<16; i++){
    a[i] = i+1.0; 
  }
   
  float *a_d, *b_d;
  cudaMalloc((void **) &a_d, 16 * sizeof(float));
  cudaMalloc((void **) &b_d, sizeof(float));

  cudaMemcpy(a_d, a, 16*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b, sizeof(float), cudaMemcpyHostToDevice);
  sumReductionKernel<<<4*4, 4>>>(a, b);

  cudaMemcpy(b, b_d, sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i<16; i++){
    printf("%.2f", a[i]);  
       
  }
       printf("%.2f", b);
}
