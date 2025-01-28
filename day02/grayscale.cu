__global__
void colorToGrayScaleConvertion(unsigned char * Pout, unsigned char * Pin, int width , int height){

  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row<height){
    int grayOffset = row * width + col;

    int rgbOffset = grayOffset * CHANNELS ;
    
    unsigned char r = Pin[rgbOffset];
    unsigned char g = Pin[rgbOffset + 1];
    unsigned char b = Pin[rgbOffset + 2];

    Pout[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
  }
}

int main(){
  dim3 dimGrid(ceil(m/16.0), ceil(n/16.0), 1);
  dim3 dimBlock(16,16,1);
  const int CHANNEL = 3;
  colorToGrayScaleConvertion<<<dimGrid, dimBlock>>>(Pout, Pin, n, m);

}
