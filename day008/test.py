import math
import os

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"

fa2_forward= load(name="fa2_forward", sources = ["bind.cpp", "flash_attention_2.cu"], extra_cuda_cflags=["-O3"])

batch_size = 1
num_heads = 1
seq_len = 2
head_dim = 3

q = torch.randn((batch_size, num_heads, seq_len, head_dim), device="cuda") 
k = torch.randn((batch_size, num_heads, seq_len, head_dim), device="cuda")
v = torch.randn((batch_size, num_heads, seq_len, head_dim), device="cuda")

def manual_torch_attention(q: torch.Tensor, k:torch.Tensor, v: torch.Tensor):
    att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(head_dim))
    att = F.softmax(att, dim=-1)
    attention = att @ v
    return attention

def benchmark(func, *args, name="Function"):
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    stream = torch.cuda.current_stream()

    start.record(stream=stream)
    result = func(*args)
    end.record(stream=stream)

    torch.cuda.synchronize()

    elapsed_time = start.elapsed_time(end)
    print(f"{name} execution time: {elapsed_time:3f} ms") 
    return result, elapsed_time


def main():
    print("Starting the benchmarking...")
    kernel_result, kernel_exec_time = benchmark(fa2_forward.fa2_forward, q, k, v, name="FA2 forward kernel") 
    kernel_res_O, kernel_res_M = kernel_result   
    manual_result = manual_torch_attention(q, k, v)
    tolerance = 1e-2
    allclose = torch.allclose(kernel_res_O, manual_result, rtol=0, atol=tolerance)
    print(f"Kernel execution time: {kernel_exec_time} ms\n")
    print(f"Values under tolerance limit in attention: {tolerance}:{allclose}")
    
    if not allclose:
        print("====================================================")
        print(f"Manual result: {manual_result}")
        print()
        print(f"Kernel result: {kernel_res_O}")

if __name__ == "__main__":
    main()
