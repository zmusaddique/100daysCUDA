#define FILTER_RADIUS 2
#define IN_TILE_DIM 32
#define OUT_TILE_DIM (IN_TILE_DIM - (2 * FILTER_RADIUS))

// use cudaMemcpyToSymbol for const mem
__constant__ float F[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];
// caching with const mem happens in hardware

__global__ void convolution_basic(float *N, float *F, float *P, int r,
                                  int width, int height) {
  int outCol = blockIdx.x * blockDim.x + threadIdx.x;
  int outRow = blockIdx.y * blockDim.y + threadIdx.y;

  float Pval = 0.0f;

  for (int fRow = 0; fRow < (2 * r + 1); fRow++) {
    for (int fCol = 0; fCol < (2 * r + 1); fCol++) {
      int inRow = outRow - r + fRow;
      int inCol = outCol - r + fCol;
      if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
        Pval += F[fRow * r + fCol] * N[inRow * width + inCol];
      }
    }
  }
  P[outRow * width + outCol] = Pval;
}

__global__ void conv_tiled(float *N, float *F, float *P, int width,
                           int height) {
  int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x;
  int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y;

  int srcCol = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
  int srcRow = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;

  __shared__ float Ns[IN_TILE_DIM * IN_TILE_DIM];

  if (srcRow >= 0 && srcRow < height && srcCol >= 0 && srcCol < width) {
    Ns[threadIdx.y * IN_TILE_DIM + threadIdx.x] = N[srcRow * width + srcCol];
  } else {
    Ns[threadIdx.y * IN_TILE_DIM + threadIdx.x] = 0.0f;
  }
  __syncthreads();

  if (col < width && row < height) {
    float Pval = 0.0f;
    for (int fRow = 0; fRow < (2 * FILTER_RADIUS + 1); fRow++) {
      for (int fCol = 0; fCol < (2 * FILTER_RADIUS + 1); fCol++) {
        int tileCol = threadIdx.x + fCol;
        int tileRow = threadIdx.y + fRow;
        Pval += F[fRow * (2 * FILTER_RADIUS + 1) + fCol] *
                Ns[tileRow * IN_TILE_DIM + tileCol];
      }
    }
    P[row * width + col] = Pval;
  }
}
