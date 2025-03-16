Prefix sum - also know as scan;
Parallel scan is frequently used to parallelize seemingly sequential ops

If a computation is naturally described a recursion where each item in a series is defined in terms of it's previous terms, it can likely be parallelized as a parallel scan op.

[3, 1, 7, 0, 4, 1, 6, 3]
Inclusive scan give us?
[3, 4, 11, 11, 15, 16, 22, 25] 
Exclusive scan
[0, 3, 4, 11, 11, 15, 16, 22] - excludes the effect of corresponding input element - Gives the starting locations of the cut- ideal for mem locations 

void sequential_inclusive_scan(float *x, float *y, int N){
  y[0] = x[0];
  for (int i = 1; i < N; i++){
    y[1] = y[i-1] + x[i];
  }
}
