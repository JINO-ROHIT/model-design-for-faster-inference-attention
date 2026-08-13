""""
dont materialize scores = (seq, seq) instead do thid in blocks + online softmax


for q_block:
    running max m
    running softmax denominator 1
    running output acc

    for k_block, v_block:
        q_block @ k_block.T
        update softmax
        update acc
"""

import torch
import math

from common import copy_weights

block_q = 16
block_kv = 16 # same for both

class Attention:
    def __init__(self, hidden_dim, batch_size, seq_len, num_heads):
        self.o = torch.randn(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = self.hidden_dim // self.num_heads
    
    def forward(self, q, k, v):

        q = q.view(self.batch_size, self.seq_len, self.num_heads, self.head_dim)
        k = k.view(self.batch_size, self.seq_len, self.num_heads, self.head_dim)
        v = v.view(self.batch_size, self.seq_len, self.num_heads, self.head_dim)

        # (batch size, num heads, seq len, head dim)
        q = torch.transpose(q, 1, 2)
        k = torch.transpose(k, 1, 2)
        v = torch.transpose(v, 1, 2)

        q_blocks = torch.split(q, block_q, dim = 2) # block over the seq dim
        k_blocks = torch.split(k, block_kv, dim = 2)
        v_blocks = torch.split(v, block_kv, dim = 2)

        output_blocks = []

        for q_blk in q_blocks:

            q_blk_size = q_blk.shape[2]

            # you need a per block stats to be maintained
            """
            being more precise here, we are doing softmax attention per block,
            so how many max and denom do we need to track for each query? every batch * every head * every query token inside a block. its just a single scalar
            for output acc, its a vector, so we need batch * head * block size * head dim
            """
            running_max = torch.full((self.batch_size, self.num_heads, q_blk_size), -float("inf"))
            softmax_denom = torch.zeros((self.batch_size, self.num_heads, q_blk_size))
            output_acc = torch.zeros((self.batch_size, self.num_heads, q_blk_size, self.head_dim))

            for k_blk, v_blk in zip(k_blocks, v_blocks):

                i = (q_blk @ torch.transpose(k_blk, 3, 2)) * math.sqrt(1/self.head_dim) # (bs, num heads, block_q, block_kv)

                running_max_new = torch.maximum(running_max, i.max(dim=-1).values)

                old_scale = torch.exp(running_max - running_max_new)
                exp_i = torch.exp(i - running_max_new[..., None]) # wee need to make it 4d since i is 4d

                softmax_denom = softmax_denom * old_scale + exp_i.sum(dim=-1)
                output_acc = output_acc * old_scale[..., None] + exp_i @ v_blk
                running_max = running_max_new

            output_blocks.append(output_acc / softmax_denom[..., None])

        v = torch.cat(output_blocks, dim=2)
        v = torch.transpose(v, 2, 1)
        v = v.reshape(self.batch_size, self.seq_len, -1)

        return v @ self.o


def main():
    torch.manual_seed(0)

    batch_size = 2
    seq_len = 1024
    hidden_dim = 512
    num_heads = 16
    head_dim = 512 // 16

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
        num_heads=num_heads,
        batch_first=True,
    )
    copy_weights(reference, torch_mha)

    reference_out = reference.forward(q, k, v)
    torch_out, _ = torch_mha(q, k, v, need_weights=False)

    print("flash attention v1")
    print("custom attn shape:", tuple(reference_out.shape))
    print("torch attn shape shape:    ", tuple(torch_out.shape))
    print("max abs diff:       ", (reference_out - torch_out).abs().max().item())
    print("close:              ", torch.allclose(reference_out, torch_out, atol=1e-5))
    print()


if __name__ == "__main__":
    main()
