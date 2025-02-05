// 1. Yes, we can reduce it from the concept of tiles. Every thread responsible
//    for an output can load corresponding inputs into global memory

// 2. Work it on paper, you will find that for N * N tile global memory accesses 
//    is just reducuced by 1/N, with N/2 accesses in each phase

// 3. __syncthreads() ensures all threads are executed to this point. In the first
//    use, It ensures all required inputs are present in shared memory to be used 
//    by all threads in the block for each phase. 
//    In second use, it ensures all threads perform calculations for that phase.
//    Because these resources will be overritten in next phase.
//    All of this just because SMs execute threads in parallel but not guarantee
//    all threads will have same state at each point in time.

// 4. Registers are local to a thread. This is not desirable in cases like tiling 
//    where threads collaborate to share the same resource to be used by threads.
//    If data-to-be-shared is stored in registers each thread should have to 
//    make a copy of its resources leading to redundant work.

// 5. Reduction is 1/32 times.

// 6. A local variable is reserved only for a thread. There will be a copy for 
//    each thread. 1000 blocks * 512 threads = 512,000 copies throughout kernel lifetime.

// 7. If variable is declared in shared memory, it will be shared across all 
//    threads in a block. ie. one copy be block. In total 1000 copies.

// 8. When no tiling, once for each operation. ie. N times each element from a matrix
//    b. In case of tiling, once for each phase  ie. N/T times for each element from a matrix

// 9. Arithmetic intensity = FLOPS/BYTES = 36/(7*(32/8)) = 1.2875
// a. Memory-bound because 1.2875 Flops/B * 100GB/s  = 128.7 GFlops/s > 100GB/s 
// b. Compute-bound because 1.2875 Flops/B * 250GB/s  = 321.43 GFlops/s > 300GFlops/s

// 10. a. The kernel will work only value 1 because of the race condition in accessing 
//      pattern of the shared memory. 
//     b. To fix the issue add __syncthreads() after setting the shared memory. Now 
//      kernel will work for all values of BLOCK_SIZE 

// 11. a. 8 * 128 = 1024 copies ie. 1/thread
//     b. 8 * 128 = 1024 versions ie, 1 per thread
//     c. 8 versions ie. 1/block
//     d. 8 versions ie. 1/block
//     e. 1*4 + (128*4) = 516 bytes
//     f. 10Flops/5*4Bytes = .5Flops/s

// 12.a. Limited by shared memory (75% occupancy)
//    b. Achieves full occupancy (no limiting factor). 
