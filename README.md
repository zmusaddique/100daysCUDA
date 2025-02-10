# Project progress and Log

This is the log of 100 Days of CUDA challenge and what I implemented during this challenge.

Mentor: https://github.com/hkproj

## Task list

| Day | Task Description                                                                                                                                | STATUS  |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| D15 | **Mandatory FA2-Forward Pass**: Implement forward pass for FA2                                                                                  | PENDING |
| D20 | **Mandatory FA2-Backward Pass**: Implement backward pass for FA2                                                                                | PENDING |
| D20 | **Side Quest Chunked Cross Entropy Loss**: Fuse the logits layer and the computation of the CE loss by chunks. (Ref. Liger Kernel imp in triton | PENDING |

## Short summary

| Day   | Files                                                                                                                                      |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| day01 | **vecAdd.cu**: Parallel vector addition <br> **answers.cu**: Answers to PMPP Chap 2                                                        |
| day02 | **matrixMult.cu**: Matrix multiplication kernel <br> **grayscale**: Color to grayscale kernel <br> **imageBlur.cu**: Blur image kernel     |
| day03 | **answers.cu**: Answers to exercise of ch3 of PMPP                                                                                         |
| day04 | **simpleSumReductionKernel.cu**: tree-based sum reduction <br> Learnings: barrier syncronization                                           |
| day05 | **convergentSumReduction.cu**: convergence to previous reduction <br> Log: Exercises of ch4                                                |
| day06 | **tiledMatMul.cu**: Tiled Matrix Multiplication                                                                                            |
| day07 | **convoluton_2d.cu**: Implemented a simple 2D convolution                                                                                  |
| day08 | **convolution_with_caching.cu** Implemented 2D convolution with tiling and caching in constant memory                                      |
| day09 | **matmulEnhanced.cu**: Enhanced the tile matrix multiplication for generalization with dynamic 1D shared memory array and memory colescing |
| day10 | **ch5_exercises.cu**: Solutions to chapter of PMPP <br> **tile_matrix_transpose.cu**: Tiled matrix transpose kernel                        |
| day11 | **convolution_2d.cu**: tiled convolution                                                                                                   |
| day12 | **convolution.cu**: tiled convolution with cached halo cells                                                                               |

# Summary

## Day 08

Enhanced the 2D convolution to implement caching and tiling.
Key points in learning:

- Intrinsic hardware caching in constant memory by `__constant__`
- shared memory

## Day 09

Enhanced the 2D Matrix mulitplication by adding dynamic shared memory and generalization (any dimensions supported).
Key Takeaways from experiments:

- Profiling tracks kernel hardware performance `ncu <executable>`
- Coalescing memory for better memory througput (use consecutive memory instead of scattered/strided memory accesses)
- Prevent garbage value errors by boundary conditions for arbitrary dimensions
- Appropriate tile size can bring drastic changes. Observations from running on colab's T4:

| Tile size | Time Taken                                                                                   |
| --------- | -------------------------------------------------------------------------------------------- |
| 2         | Non-tiled kernel execution time: 41609.312 ms <br> Tiled kernel execution time: 99787.430 ms |
| 4         | Non-tiled kernel execution time: 16879.109 ms <br> Tiled kernel execution time: 17574.977 ms |
| 8         | Non-tiled kernel execution time: 8604.168 ms <br> Tiled kernel execution time: 5561.509 ms   |
| 16        | Non-tiled kernel execution time: 5727.267 ms <br> Tiled kernel execution time: 4158.605 ms   |
| 32        | Non-tiled kernel execution time: 4160.248 ms <br> Tiled kernel execution time: 4791.448 ms   |
| 64        | Non-tiled kernel execution time: 0.826 ms <br> Tiled kernel execution time: 0.347 ms         |
| 128       | Non-tiled kernel execution time: 0.838 ms<br> Tiled kernel execution time: 0.238 ms          |

## Day 10

Implemented tiled matrix transpose kernel
Solved exercises of Chapter 5 from PMPP

Key learnings:

- Optimize for occupancy
- Check for compute / memory boundedness in applications
- Improve arithmetic intensity
- Look for race conditions among threads in a block in shared memory access patterns

## Day 11

Added tiling to 2D convolution kernel

## Day 12

Added caching for halo cells in 2D convolution

Key learnings

- Constant Memory
- L1, L2, L3 Cache
