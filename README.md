# Project progress and Log

This is the log of 100 Days of CUDA challenge and what I implemented during this challenge.

Mentor: https://github.com/hkproj

## Task list

| Day | Task Description                                                                                                                                |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| D15 | **Mandatory FA2-Forward Pass**: Implement forward pass for FA2                                                                                  |
| D20 | **Mandatory FA2-Backward Pass**: Implement backward pass for FA2                                                                                |
| D20 | **Side Quest Chunked Cross Entropy Loss**: Fuse the logits layer and the computation of the CE loss by chunks. (Ref. Liger Kernel imp in triton |

## Short summary 

| Day   | Files                                                                                                                         |
| ----- | -------------------------------------------------------------------------------------------------------------------------------------- |
| day01 | **vecAdd.cu**: Parallel vector addition <br> **answers.cu**: Answers to PMPP Chap 2                                                    |
| day02 | **matrixMult.cu**: Matrix multiplication kernel <br> **grayscale**: Color to grayscale kernel <br> **imageBlur.cu**: Blur image kernel |
| day03 | **answers.cu**: Answers to exercise of ch3 of PMPP                                                                                     |
| day04 | **simpleSumReductionKernel.cu**: tree-based sum reduction <br> Learnings: barrier syncronization                                       |
| day05 | **convergentSumReduction.cu**: convergence to previous reduction <br> Log: Exercises of ch4                                            |
| day06 | **tiledMatMul.cu**: Tiled Matrix Multiplication                                                                                        |
| day07 | **convoluton_2d.cu**: Implemented a simple 2D convolution                                                                              |
| day08 | **convolution_with_caching.cu** Implemented 2D convolution with tiling and caching in constant memory                                  |

# Summary

## Day 08
Enhanced the 2D convolution to implement caching and tiling. 
Key points in learning:
- Intrinsic hardware caching in constant memory by `__constant__`
- shared memory 
 
