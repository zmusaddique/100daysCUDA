### Log

## Update after today's work

Backward pass is now fully functional and complete for both causal and non-causal attention!

#### Current Status

**dQ is computed successfully for both causal and non-causal attention.**

With the current state, there are a few things that can now be considered correct (to avoid massive debugging times):

- dQ is correctly computed, meaning all dependent variables on dQ must be correct.
- This implies that dS, P, dP, dO, and D must also be correct, which has been verified.

Mathematically, we have:

$$
dV = P^T dO
$$

$$
dK = dS \cdot Q
$$

Should be a small thing now.

I noticed something

Instead of this:

```python
for (int q = threadIdx.y; q < Br && (q_tile_idx + q) < seq_len; q += blockDim.y) {
  for (int k = threadIdx.x; k < Bc && (k_tile_idx + k) < seq_len; k += blockDim.x) {
    P_g[batch_head_idx * seq_len * seq_len + (q_tile_idx + q) * seq_len +
        (k_tile_idx + k)] += P[q * Bc + k];
  }
}
```

do this

```python
const int base_Idx_2 = batch_head_idx * seq_len * seq_len;
    for (int q = threadIdx.y; q < Br && (q_tile_idx + q) < seq_len;
         q += blockDim.y) {
      const int q_idx = q_tile_idx + q;
      const int q_offset = q_idx * seq_len;
      for (int k = threadIdx.x; k < Bc && (k_tile_idx + k) < seq_len;
           k += blockDim.x) {
        const int k_idx = k_tile_idx + k;
        P_g[base_Idx_2 + q_offset + k_idx] += P[q * Bc + k];
      }
    }
    __syncthreads();
```

This will save per thread register usage

I get error in dK. The outputs are similar to the test but maybe in a wrong scale.

dK's dependent variables are dS and Q. dS is correct. That should mean there might be a bug in accessing dS and Q.

After 1 hr of debugging and prompting: the computation of tiled dK is legit correct, however problem is how these dK_tiled is accumulated to dK (global). chatGPT gave suggested acumulating only when threadIdx.y = 0.

[Redundant indexing](/media/ridiculous.png)
It works! for both causal and non-causals! but why? I am trying to understand this. I iterate for every d in tx. This is bad because I am not leveraging parallelism and other threads.

[Not so redundant](/media/sensible.png)
Turns out that I am indeed writing to global memory wrong. A fix in the looping helped and with this I am mapping to unique indices

[Success](/media/happiness.png)
:)
