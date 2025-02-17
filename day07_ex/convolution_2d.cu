__global__ void convolution2D(float *A, float *F, float *R, int r, int width, int height){
  outputCol = blockIdx.x * blockDim.x + threadIdx.x;
  outputRow = blockIdx.y + blockDim.y + threadIdx.y;

  float Pval = 0.0;
  for (int fRow = 0; fRow < 2*r+1; fRow++){
    for (int fCol = 0; fCol < 2*r + 1; fCol++){
      inRow=outputRow - r + fRow;
      inCol = outputCol - r + fCol;
      if (inRow >= 0 && inRow < height && inCol >=0 && inCol <width){ 
      Pval += F[fRow][fCol] * N[inRow * width + inCol];   
      }
    } 
  }
}  
