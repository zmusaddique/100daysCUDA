#define FILTER_RADIUS 2
#define IN_TILE_DIM 32
#define OUT_TILE_DIM (IN_TILE_DIM - 2*(FILTER_RADIUS))

__constant__ float F[2*FILTER_RADIUS + 1][2*FILTER_RADIUS +1]

__global__ void convolution_with_const_mem(float *A, float *R, int r, int width, int height) {
  int outRow = blockIdx.y * blockDim.y + threadIdx.y;   
  int outCol = blockIdx.x * blockDim.x + threadIdx.x;

  float Rval = 0.0;
  for (int i = 0; i<2*r+1; i++){
    for(int j=0; j<2*r+1; j++){
      int inRow = outRow - r + i;
      int inCol = outCol - r + j;
      if (inRow>=0 && inRow < height && inCol>=0 && inCol < width){
        Rval += F[i][j] * A[inRow * width + inCol]; 
      }
    } 
  }
    
  R[outRow * width + outCol] = Rval; 
}

__global__ void convolution_with_tiling(float *A, float*R, int width, int height){
  int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS; 
  int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;

  __shared__ A_s[IN_TILE_DIM][IN_TILE_DIM];

  if(row>=0 && row<height && col>=0 && col<width){
    A_s[threadIdx.y][threadIdx.x] = A[row*width + col];    
  }else {
    A_s[threadIdx.y][threadIdx.x]=0.0;
  }

  int tileRow = threadIdx.x - FILTER_RADIUS;
  int tileCol = threadIdx.y - FILTER_RADIUS;

  __syncthread();

  if(row>=0 && row<height && col>=0 && col<width){
    if(tileRow>=0 && tileRow<OUT_TILE_DIM && tileCol >= 0 && tileCol<OUT_TILE_DIM){
    float Rval = 0.0;
    for (int fRow=0; fRow < 2 * FILTER_RADIUS +1; fRow++){
      for (int fcol=0; fCol < 2 * FILTER_RADIUS +1; fCol++){
        Rval += F[fRow][fCol] * A_s[tileRow + fRow][tileCol +fCol];  
      }
    }
    Rval[row * width + col]= Rval;
    }
  }
}  
