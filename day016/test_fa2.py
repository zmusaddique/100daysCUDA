import os

import torch

from torch.utils.cpp_extension import load

os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"

fa2_attention = load(name="fa2_attention", sources = ["bind.cpp", "fa2_complete.cu"], extra_cflags=["-g", "-O3"], extra_cuda_cflags=["-g", "-G", "-lineinfo"])


def manual_attention(Q, K , V, causal=False):
    batch_size, num_heads, seq_len, head_dim = Q.shape
    softmax_scale  = 1.0 /  (head_dim ** 0.5)

    scores = torch.matmul(Q, K.transpose(-1, -2)) * softmax_scale

    if causal:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=Q.device)).bool()
        scores = scores.masked_fill(~mask, float("-inf"))

    P = torch.softmax(scores, dim=-1)
    O = torch.matmul(P, V)

    return O

def manual_attention_backward(dO, Q, K, V, O, causal=False):
    batch_size, num_heads, seq_len, head_dim = Q.shape
    softmax_scale = 1.0 / (head_dim ** 0.5)

    scores = torch.matmul(Q, K.transpose(-1, -2)) * softmax_scale
    if causal:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=Q.device)).bool()
        scores = scores.masked_fill(~mask, float('-inf'))

    P = torch.softmax(scores, dim=-1)

    dV = torch.matmul(P.transpose(-1, -2), dO)
    dP = torch.matmul(dO, V.transpose(-1, -2))
    D = torch.sum(O * dO, dim=-1, keepdim=True)
    dS = P * (dP - D)
    dQ = torch.matmul(dS, K)
    dK = torch.matmul(dS.transpose(-1, -2), Q)

    return dQ, dK, dV

def test_fa2_attention(batch_size, num_heads, seq_len, head_dim, causal):
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', requires_grad=True)
    K =  torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', requires_grad=True)
    V =  torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', requires_grad=True)
    dO =  torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')

    O_manual = manual_attention(Q, K, V, causal)

    O_cuda, M_cuda = fa2_attention.forward(Q, K, V, causal)

    dQ_backward, dK_backward, dV_backward = manual_attention_backward(dO, Q, K, V, O_manual, causal)

    dQ_cuda, dK_cuda, dV_cuda = fa2_attention.backward(dO, Q, K, V, O_cuda, M_cuda, causal)

    atol = 1e-5
    rtol = 1e-5

    print(f"\nTesting with causal={causal}")
    print("Forward output comparison: ")
    print(f"Max diff: {torch.max(torch.abs(O_manual - O_cuda)):.8f}")
    assert torch.allclose(O_manual, O_cuda, atol=atol, rtol=rtol), "Forward outputs differ!"

    print("dQ comparison: ")
    print(f"Max diff: {torch.max(torch.abs(dQ_backward-dQ_cuda))}")
    assert torch.allclose(dQ_backward, dQ_cuda, atol=atol, rtol=rtol), "dQ differs!"

    print("dK comparison: ")
    print(f"Max diff: {torch.max(torch.abs(dK_backward-dK_cuda))}")
    assert torch.allclose(dK_backward, dK_cuda, atol=atol, rtol=rtol), "dK differs!"

    print("dV comparison: ")
    print(f"Max diff: {torch.max(torch.abs(dV_backward-dV_cuda))}")
    assert torch.allclose(dV_backward, dV_cuda, atol=atol, rtol=rtol), "dV differs!"

    print("Test passed!")

if __name__ == "__main__":
    test_fa2_attention(batch_size=2, num_heads=4, seq_len=32, head_dim=64,causal=False)
    test_fa2_attention(batch_size=2, num_heads=4, seq_len=32, head_dim=64,causal=True)
