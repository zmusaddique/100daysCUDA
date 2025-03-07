import os

import torch
from torch.utils.cpp_extension import load

os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"

fa2_attention = load(name="fa2_attention", sources = ["bind.cpp", "fa2_complete.cu"], extra_cflags=["-g", "-O3"], extra_cuda_cflags=["-g", "-G", "-lineinfo"])


def manual_attention(Q, K , V, causal=False):
    _,_, seq_len, head_dim = Q.shape
    softmax_scale  = 1.0 /  (head_dim ** 0.5)

    scores = torch.matmul(Q, K.transpose(-1, -2)) * softmax_scale

    if causal:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=Q.device)).bool()
        scores = scores.masked_fill(~mask, float("-inf"))

    M_ref = scores.max(dim=-1)[0]
    S_ref = torch.exp(scores - M_ref.unsqueeze(-1))
    L_ref = S_ref.sum(dim=-1)
    P_ref = S_ref/ L_ref.unsqueeze(-1)
    O = torch.matmul(P_ref, V)

    return O

def manual_attention_backward(dO, Q, K, V, O, causal:bool):
    _,_, seq_len, head_dim = Q.shape
    softmax_scale = 1.0 / (head_dim ** 0.5)

    scores = torch.matmul(Q, K.transpose(-1, -2)) * softmax_scale
    if causal:
        # scores = scores.clone()
        mask = torch.tril(torch.ones(seq_len, seq_len, device=Q.device)).bool()
        scores = scores.masked_fill(~mask, float('-inf'))
        # torch.cuda.synchronize()
    P = torch.softmax(scores, dim=-1)

    dV = torch.matmul(P.transpose(-1, -2), dO)
    dP = torch.matmul(dO, V.transpose(-1, -2))
    D = torch.sum(dO * O, dim=-1, keepdim=True)
    dS = P * (dP - D)
    dQ = torch.matmul(dS, K) * softmax_scale
    dK = torch.matmul(dS.transpose(-1, -2), Q)

    return dQ, D, P

def test_fa2_attention(Q, K, V, dO, batch_size, num_heads, seq_len, head_dim, causal):
    
    O_manual = manual_attention(Q, K, V, causal)

    O_cuda, M_cuda= fa2_attention.forward(Q, K, V, causal)

    dQ_manual, D_manual, P_manual = manual_attention_backward(dO, Q, K, V, O_manual, causal)

    dQ_cuda, D_cuda, P_cuda = fa2_attention.backward(dO, Q, K, V, O_cuda, M_cuda, causal)

    atol = 1e-5
    rtol = 1e-5

    print(f"\nTesting with causal={causal}")    

    print("Forward output comparison: ")
    print(f"dO Max diff: {torch.max(torch.abs(O_manual - O_cuda)):.8f}")
    assert torch.allclose(O_manual, O_cuda, atol=atol, rtol=rtol), "Forward outputs differ!"

    # print("M comparison: ") 
    # print(f"Max diff M: {torch.max(torch.abs(M_manual - M_cuda)):.8f}")
    # assert torch.allclose(M_manual, M_cuda, atol=atol, rtol=rtol), "M differs!"
    #
    # print("L comparison:")
    # print(f"Max diff L: {torch.max(torch.abs(L_manual - L_cuda))}")
    # assert torch.allclose(L_manual, L_cuda, atol=atol, rtol=rtol), "L differs!"
    #
    # print("S comparison:")
    # print(f"Max diff S: {torch.max(torch.abs(S_manual - S_cuda))}")
    # assert torch.allclose(S_manual, S_cuda, atol=atol, rtol=rtol), "S differs!"
    #
    print("P comparison:")
    print(f"Max diff P: {torch.max(torch.abs(P_manual - P_cuda))}")
    if not torch.allclose(P_manual, P_cuda, atol=atol, rtol=rtol):
      print(f"P_manual: {P_manual}\nP_cuda: {P_cuda}")
    assert torch.allclose(P_manual, P_cuda, atol=atol, rtol=rtol), "P differs!"
   
    # print("D comparison:")
    # D_manual_squeezed = D_manual.squeeze(-1)
    # assert D_manual_squeezed.shape == D_cuda.shape, f"D_manual.shape = {D_manual_squeezed.shape}\nD_cuda.shape = {D_cuda.shape}"
    # print(f"Max diff D: {torch.max(torch.abs(D_manual_squeezed - D_cuda))}")
    # assert torch.allclose(D_manual_squeezed, D_cuda, atol=atol, rtol=rtol), "D differs!"

    # print("dS comparison:")
    # print(f"Max diff dS: {torch.max(torch.abs(dS_manual - dS_cuda))}")
    # if torch.allclose(dS_manual, dS_cuda, atol=atol, rtol=rtol):
    #     print(f"dS_manual: {dS_manual}\ndS_cuda: {dS_cuda}")
    # assert torch.allclose(dS_manual, dS_cuda, atol=atol, rtol=rtol), f"dS differs! dS_manual: {dS_manual}\ndS_manual.shape: {dS_manual.shape}\n\ndS_cuda: {dS_cuda}\ndS_cuda.shape: {dS_cuda.shape}"
    #
    print("dQ comparison:")
    print(f"Max diff dQ: {torch.max(torch.abs(dQ_manual - dQ_cuda))}")
    assert torch.allclose(dQ_manual, dQ_cuda, atol=atol, rtol=rtol), "dQ differs!"
    if torch.allclose(dQ_manual, dQ_cuda, atol=atol, rtol=rtol):
      print("dQ matches")

    # print("dK comparison:")
    # print(f"Max diff dK: {torch.max(torch.abs(dK_manual - dK_cuda))}")
    # assert torch.allclose(dK_manual, dK_cuda, atol=atol, rtol=rtol), "dK differs!"
    #
    # print("dV comparison:")
    # print(f"Max diff dV: {torch.max(torch.abs(dV_manual - dV_cuda))}")
    # assert torch.allclose(dV_manual, dV_cuda, atol=atol, rtol=rtol), "dV differs!"
    #
    print("Test passed!")

if __name__ == "__main__":
    batch_size=1
    num_heads=1
    seq_len=32
    head_dim=64

    Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', requires_grad=True)
    K =  torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', requires_grad=True)
    V =  torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', requires_grad=True)
    dO =  torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')

    test_fa2_attention(Q, K, V, dO, batch_size=2, num_heads=4, seq_len=32, head_dim=64,causal=False)
    test_fa2_attention(Q, K, V, dO,batch_size=2, num_heads=4, seq_len=32, head_dim=64,causal=True)
