__global__
void blurImageKernel(unsigned char * in, unsigned char * out, int w, int h){
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col < w && row < h){
    int pixVal = 0;
    int pixels= 0;

    for (int blurRow=-BLUR_SIZE;blurRow<BLUR_SIZE+1;++blurRow){ // computes the values for blurring a single pixel
      for(int blurCol=-BLUR_SIZE;blurCol<BLUR_SIZE+1; ++blurCol){
        int curRow = row + blurRow;
        int curCol = col + blurCol;

        if (curRow >=0 && curRow < h && curCol>=0 && curCol<w){
          pixVal += in[curRow*w+curCol];
          ++pixels;
        }

      } 
    }
    out[row * w + col] = (unsigned char)(pixVal/pixels); // writing the blur 
  }

}
