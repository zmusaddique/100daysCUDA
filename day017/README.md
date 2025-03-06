### Log

It is still a buggy kernel

What are the issues?

- dQ is calculated wrong hugely - The access patterns seem right and also introduced smem padding. The computation operations also seems right. There might be a basic issue that I'm overlooking. I need to analyze the inputs and the overall functionality of compute_dQ.

What are the ops performed by compute_dQ? - these equations feel correcly implemented
What are the inputs to compute_dQ? Are they erroneous? - The inputs are Q, K, V, dO, M, D, dQ, softmax_scale, batch_size, num_heads, seq_len, head_dim, Br, Bc, causal - None seem dependent on any kernel of backward pass

Wait, are these normalized in the first place?
I'm doing dS[q * Bc + k] = S[q * Bc + k] \* (dP - Di) - But where's P? This the softmax applied to S! I just compute the exp(of S but not normalize it)

One way is the pass the sum from the forward pass, but let's try computing it seperately. (I know it's additional compute and code)

Let's try implement sum_S, forget it I'll use the existing code

There is some discrepency still unsolved. I'm rewriting the dQ kernel. After revisiting the math in paper, I found that I don't need to compute and store S. I can use $$= \sum_{j} e^{q_i^T k_j} L_i (do_i^T v_j - D_i) k_j$$, effectively just using the existing variables. Thought I'd finish this today :|
