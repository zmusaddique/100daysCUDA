#### Log

Topics covered

- Coarsening threads to reduce the price for parallelism out of hardware capacity
- Aggregation of values in cases of pixels with large concentrations of a small set of values.
  We use a register to accumulate the values concentrated on a single place, then use atomicAdd to update with serialization.
