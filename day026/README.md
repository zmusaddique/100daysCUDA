#### Log

Topics covered

- Minimizing the memory divergence - simple stride adjustment resulting in coalesced access resulting in significantly lower global memory requests, reducing divergence of warps. (later divergence is only in the last remaining warp for about last 5 iterations)
- Minimizing global memory accesses - previous implementation
- Hierarchical reduction for arbitrary input length - Breakfree the constraint of computing in single block. update value at last from all blocks using atomicAdd.
- Coarsening - reduce the parallelism penalty by implementing an efficient serialization by COARSE_FACTOR
