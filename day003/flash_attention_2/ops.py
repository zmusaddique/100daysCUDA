import torch
from torch import Tensor

__all__ = ["fa2_forward"]

def fa2_forward(Q: Tensor, K: Tensor, V: Tensor, O: Tensor, L: Tensor, N: int, d: int, Tr: int, Tc: int):
    """Flash attention 2"""
    return torch.ops.flash_attention_2.flash_attention_fwd.default(Q, K, V, O, L, N, d, Tr, Tc) 
