"""
regular ol attention

i = (q @ k^T) / sqrt(d)
p = softmax(i)
v = p @ v
linear = v @ o
"""

import torch
import math

from common import copy_weights

class Attention:
    def __init__(self, hidden_dim):
        self.o = torch.randn(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim
    
    def forward(self, q, k, v):
        i = (q @ torch.transpose(k, 2, 1)) * math.sqrt(1/self.hidden_dim)
        p = torch.softmax(i, dim=-1)
        v = p @ v

        # print(v.shape)
        # print((v @ self.o).shape)
        return v @ self.o


def main():
    torch.manual_seed(0)

    batch_size = 2
    seq_len = 4
    hidden_dim = 8

    q = torch.randn(batch_size, seq_len, hidden_dim)
    k = torch.randn(batch_size, seq_len, hidden_dim)
    v = torch.randn(batch_size, seq_len, hidden_dim)

    reference = Attention(hidden_dim)
    torch_mha = torch.nn.MultiheadAttention(
        embed_dim=hidden_dim,
        num_heads=1,
        batch_first=True,
    )
    copy_weights(reference, torch_mha)

    reference_out = reference.forward(q, k, v)
    torch_out, _ = torch_mha(q, k, v, need_weights=False)

    print("single head attention")
    print("custom attn shape:", tuple(reference_out.shape))
    print("torch attn shape shape:    ", tuple(torch_out.shape))
    print("max abs diff:       ", (reference_out - torch_out).abs().max().item())
    print("close:              ", torch.allclose(reference_out, torch_out, atol=1e-6))
    print()


if __name__ == "__main__":
    main()
