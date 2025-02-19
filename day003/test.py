import math
import os

# import flassh_attention_2.ops
import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"


fa2_fwd = load(name="fa2_forward", sources=["build.cpp", "flash_attention_2.cu"], extra_cuda_cflags=["-O3"])

batch_size = 16
n_head = 8
seq_len = 512
head_embd_dim = 64

q = torch.randn((batch_size, n_head, seq_len, head_embd_dim), device="cuda")
k = torch.randn((batch_size, n_head, seq_len, head_embd_dim), device="cuda")
v = torch.randn((batch_size, n_head, seq_len, head_embd_dim), device="cuda")


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


print(
    f"Batch size: {batch_size},\nNumber of heads: {n_head},\nSequence length: {seq_len},\nHead dimension: {head_embd_dim}\n"
)

print("benchmarking manual attention...")
manual_result, manual_attention_time = benchmark(
    manual_attention, q, k, v, name="Manual attention from PyTorch"
)


print(f"\nbenchmarking custom cuda attention...")
custom_result, custom_attention_time = benchmark(fa2_fwd.fa2_forward, q, k, v, name="Custom Attention")


print("speed up")
speed_up = manual_attention_time / custom_attention_time
print(f"Custom CUDA attention {speed_up}x faster than manual attention")


print("Accuracy check")
tolerance = 1e-2
allclose = torch.allclose(custom_result, manual_result, rtol=0, atol=tolerance)
print(f"Attention scores within tolerance: ({tolerance}: {allclose})")
