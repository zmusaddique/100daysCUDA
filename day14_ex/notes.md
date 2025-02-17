Key topics involved:

- Multi head attention -> optimize the softmax operations
- Flash attention -> How can we block the matrix multiplications
- Make softmax safe -> Prevent exploding exponents
- Online softmax -> Heck, we proved it using induction
- Block matmuls -> Resulting in blocks of matrices, not scalars


Things going on in head
- We compute the matMUl in parallel by blocking it.
- We now need to aggregate the individual computations to match global maxima(global context)
- For this we need to fix it.

