// Chapter 4
// 
// 1. 
// a. Considering warp size of 32 threads.
// a. warps per block = No. of threads in a block / 32 = 128/32 = 4
// 
// b. warps per grid = No. of threads in grid / 32 = (8 * 128) / 32 = 1024 / 32 = 32  
// 
// c. Total threads in block = 128
//    Total threads in grid = 1024
//    Total inactive threads = 103-40 + 1 = 64 // +1 Because of 0 starting index and 
//    Total active threads = 1024 - 64 = 960 
//   i. How many warps in the grid are active?  
//     Total active warps = 960 / 32 = 30 
//  
//  ii. How many warps in the grid are divergent?
//      0-31 => Not divergent
//      32-63 => 32-39 ND; 40-63 D;
//      64-95 > 64-95 ND;
//      96-127 => 96-103 D; 104-127 ND;
//     A. 2 
//  iii. 100% 
//   iv. 8/32 = 25%
//    v. 24/32 = 75%
// d. 
//  i. 32 because all contain even Idxs
// ii. 32 because all containg even and odd idxs
// iii. 50%
// 
// e. 2 iterations
//    3 iterations
// 
// 2. 512 * 4 = 2048
// 
// 3. 1 warp
// 
// 4. 4.1 ms
// 
// 5. All threads must execute / encounter the same barrier
// 
// 6. c. 512 Because  512 * 4 = 1536 threads

// 7. a, 50%
//    b, 50%
//    c, 50%
//    d, 100%
//    e, 100%

// 8. a. 128 * 30 * 16= 61440
//  61440/65536 = 93% occupancy
// 
//
//   b. 32 * 32 = 1024 threads => 50% 
//     Threads / block

//  c. 256 * 9 = 2304 * 34 reg = 78336 regs
//      Threads / block

// 9. 32 * 32 = 1024 threads / block;
//    Better don't lie student :)  

    
