import math
import os
import sys

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"


fa2_fwd = load(name="fa2_forward", sources=["build.cpp", "flash_attention_2.cu"], extra_cuda_cflags=["-O3"])

batch_size = 1
n_head = 1
seq_len = 2 
head_embd_dim = 3 

q = torch.ones((batch_size, n_head, seq_len, head_embd_dim), device="cuda") * 0.1 + 0.5
k = torch.ones((batch_size, n_head, seq_len, head_embd_dim), device="cuda") * 0.1 + 0.5
v = torch.ones((batch_size, n_head, seq_len, head_embd_dim), device="cuda") * 0.1 + 0.5


def manual_attention(q, k, v):
    attention = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
    attention = F.softmax(attention, dim=-1)
    y = attention @ v
    return y


def benchmark(func, *args, name="Funktion"):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    result = func(*args)
    end.record()

    torch.cuda.synchronize()
    elapsed_time = start.elapsed_time(end)
    print(f"{name} execution time: {elapsed_time:3f} ms")

    return result, elapsed_time


def debug_outputs(manual, custom, name="Attention"):
    print(f"\n---- {name} ----")
    
#
# print(
#     f"Batch size: {batch_size},\nNumber of heads: {n_head},\nSequence length: {seq_len},\nHead dimension: {head_embd_dim}\n"
# )

def debug_q_kt_mul(q, k):   
    print("Q * K^T: ")
    Q_Kt_result , Q_Kt_time = benchmark(fa2_fwd.q_kt_mul, q, k, name="Q_KT MatMul")
    Q_Kt_Pytorch = q @ k.transpose(-2,-1)
    tolerance = 1e-2
    allclose = torch.allclose(Q_Kt_result, Q_Kt_Pytorch, rtol=0, atol=tolerance)
    print(f"Values under tolerance for Q_K^T mul: ({tolerance}: {allclose}) ") 

def main(debug=False):
    if debug:
        q = torch.tensor([[[[1.0, 2.0],[0.5, 1.0],[0.0, 2.0]]]]).cuda()
        k = torch.tensor([[[[2.0, 0.0],[1.0, 1.0],[0.0, 2.0]]]]).cuda()
        print("Running in DEBUG mode")
        debug_q_kt_mul(q, k)
    else:
        print("Running in NORMAL mode")

if __name__ == "__main__":
    debug_mode = '--debug' in sys.argv

    main(debug=debug_mode)
# print("benchmarking manual attention...")
# manual_result, manual_attention_time = benchmark(
#     manual_attention, q, k, v, name="Manual attention from PyTorch"
# )
# print("Manual computed attention: ", manual_result)
#
# print(f"\nbenchmarking custom cuda attention...")
# custom_result, custom_attention_time = benchmark(fa2_fwd.fa2_forward, q, k, v, name="Custom Attention")
# print("Custom attention result: ",custom_result)
#
# print("speed up")
# speed_up = manual_attention_time / custom_attention_time
# print(f"Custom CUDA attention {speed_up}x faster than manual attention")
#
#
# print("Accuracy check")
# tolerance = 1e-2
# allclose = torch.allclose(custom_result, manual_result, rtol=0, atol=tolerance)
# print(f"Attention scores within tolerance: ({tolerance}: {allclose})")
