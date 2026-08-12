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
    def __init__(self, hidden_dim, batch_size, seq_len, num_heads):
        self.o = torch.randn(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = self.hidden_dim // self.num_heads
    
    def forward(self, q, k, v):
        # q = q.view(self.batch_size, self.num_heads, self.seq_len, self.head_dim) # question for you - think about whats the issue doing this
        # k = k.view(self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        # v = v.view(self.batch_size, self.num_heads, self.seq_len, self.head_dim)

        q = q.view(self.batch_size, self.seq_len, self.num_heads, self.head_dim)
        k = k.view(self.batch_size, self.seq_len, self.num_heads, self.head_dim)
        v = v.view(self.batch_size, self.seq_len, self.num_heads, self.head_dim)

        q = torch.transpose(q, 1, 2)
        k = torch.transpose(k, 1, 2)
        v = torch.transpose(v, 1, 2)

        i = (q @ torch.transpose(k, 3, 2)) * math.sqrt(1/self.head_dim) # (bs, num heads, seq, seq)
        p = torch.softmax(i, dim=-1) # (bs, num heads, seq, seq)
        v = p @ v # (bs, num heads, seq, head dim)

        #v = v.view(self.batch_size, self.seq_len, -1) # question for you - think about why this wont work

        v = torch.transpose(v, 2, 1)
        v = v.reshape(self.batch_size, self.seq_len, -1)

        return v @ self.o


def main():
    torch.manual_seed(0)

    batch_size = 2
    seq_len = 4
    hidden_dim = 8
    num_heads = 2
    head_dim = 4

    # q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    # k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    # v = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # torch will do the split internally
    q = torch.randn(batch_size, seq_len, hidden_dim)
    k = torch.randn(batch_size, seq_len, hidden_dim)
    v = torch.randn(batch_size, seq_len, hidden_dim)

    reference = Attention(hidden_dim, batch_size, seq_len, num_heads)
    torch_mha = torch.nn.MultiheadAttention(
        embed_dim=hidden_dim,
        num_heads=2,
        batch_first=True,
    )
    copy_weights(reference, torch_mha)

    reference_out = reference.forward(q, k, v)
    torch_out, _ = torch_mha(q, k, v, need_weights=False)

    print("multi head attention")
    print("custom attn shape:", tuple(reference_out.shape))
    print("torch attn shape shape:    ", tuple(torch_out.shape))
    print("max abs diff:       ", (reference_out - torch_out).abs().max().item())
    print("close:              ", torch.allclose(reference_out, torch_out, atol=1e-6))
    print()


if __name__ == "__main__":
    main()
