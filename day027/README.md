#### Log

Topics covered

- Scan algo overview
- Intrinsic and extrinsic scan
- Limitations with the sequential scan to be executed in parallel
- Kogge-stone algo - sum of 2^k elements before x[i]
- In-place scan - So intermediate results need to be handled to avoid races
- second barrier turning to be a bottle-neck;
- Double buffer approach - eliminate second barrier sync.
