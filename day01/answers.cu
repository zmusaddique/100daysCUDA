// 1. C 
// Why? Primary way of mapping block & thread index to data index


// 2. C 
// Why?: 
// i=(blockIdx.x *blockDim.x + threadIdx.x)*2 
// Take blockDim.x = 256
// For blockIdx.x = 0 & threadIdx.x = 0, i = 0
// For blockIdx.x = 0 & threadIdx.x = 1, i = 2 
// For blockIdx.x = 0 & threadIdx.x = 2, i = 4
//
// The first elements of adjacent data are correctly mapped


// 3. D 
// Why?
// i=blockIdx.x * blockDim.x * 2 + threadIdx.x;
// Take blockDim.x = 256
// For blockIdx.x = 0 & threadIdx.x = 0, i = 0
// For blockIdx.x = 0 & threadIdx.x = 1, i = 1 
// For blockIdx.x = 0 & threadIdx.x = 2, i = 2
// ...
// For blockIdx.x = 1 & threadIdx.x = 0, i = 512
//
// The entire first section is processed by block 0 (0-511)


// 4. C 
// Why ?
// Blocks = ceil(8000/1024) = 8
// Thread in each block = 1024 
// Total threads  = 8 * 1023 = 8192


// 5. D 
// Why? 
// we allocate no of bytes in malloc
// sizeof(int) - bytes for a single int; v - no of total ints
// v * sizeof (int)


// 6. D 
// Why?
// A_d is float pointer to the address
// &A_d is address of A_d and is of type float** (pointer to float*)  
// since cudaMalloc requires generic type, we cast it to (void**)


// 7. C
// Syntax - cudaMalloc(destination, source, bytes, direction)


// 8. C 
// Specified in book 


// 9. 
// a. Threads in a block: 128
// b. Threads in grid: (N+128-1)/128 * 128 = (200000+128-1)/128 * 128 = 1563*128 = 200064
// c. 1563
// d. 200064 - Because created threads work, may not compute because we added a condition
// e. 200000 - Extra threads filtered by condition 
