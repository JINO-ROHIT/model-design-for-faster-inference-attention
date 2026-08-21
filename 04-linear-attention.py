"""
linear attention

regular attention does:

scores = q @ k^T
p = softmax(scores)
out = p @ v

linear attention changes the kernel so we can rearrange the work:

out_t = phi(q_t) @ sum(phi(k_i)^T @ v_i)
        ---------------------------------------
        phi(q_t) @ sum(phi(k_i))

for causal decoding, the two sums are just running state.
so instead of storing all old keys and values, we keep:

ssm   = sum(phi(k_i)^T @ v_i)   # (head dim, head dim)
z     = sum(phi(k_i))           # (head dim)
"""

import torch


class LinearAttention:
    def __init__(self, hidden_dim, batch_size, seq_len, num_heads):
        self.o = torch.randn(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = self.hidden_dim // self.num_heads

    def feature_map(self, x):
        # Need positive features so the denominator behaves like attention weights.
        return torch.nn.functional.elu(x) + 1

    def split_heads(self, x):
        x = x.view(self.batch_size, self.seq_len, self.num_heads, self.head_dim)
        return torch.transpose(x, 1, 2)

    def merge_heads(self, x):
        x = torch.transpose(x, 2, 1)
        return x.reshape(self.batch_size, self.seq_len, self.hidden_dim)

    def forward_prefill(self, q, k, v):
        q = self.feature_map(self.split_heads(q))
        k = self.feature_map(self.split_heads(k))
        v = self.split_heads(v)

        output = []
        ssm = torch.zeros(self.batch_size, self.num_heads, self.head_dim, self.head_dim)
        z = torch.zeros(self.batch_size, self.num_heads, self.head_dim)

        for token_idx in range(self.seq_len):
            k_t = k[:, :, token_idx] # (bs, num heads, head dim)
            v_t = v[:, :, token_idx] # (bs, num heads, head dim)
            q_t = q[:, :, token_idx] # (bs, num heads, head dim)

            ssm = ssm + k_t[..., :, None] @ v_t[..., None, :]
            z = z + k_t

            numerator = q_t[..., None, :] @ ssm
            denominator = q_t[..., None, :] @ z[..., :, None]
            out_t = numerator.squeeze(-2) / denominator.squeeze(-1).clamp_min(1e-6)
            output.append(out_t)

        output = torch.stack(output, dim=2)
        output = self.merge_heads(output)
        return output @ self.o

    def forward_decode(self, q, k, v):
        q = self.feature_map(self.split_heads(q))
        k = self.feature_map(self.split_heads(k))
        v = self.split_heads(v)

        output = []
        ssm = torch.zeros(self.batch_size, self.num_heads, self.head_dim, self.head_dim)
        z = torch.zeros(self.batch_size, self.num_heads, self.head_dim)

        for token_idx in range(self.seq_len):
            k_t = k[:, :, token_idx]
            v_t = v[:, :, token_idx]
            q_t = q[:, :, token_idx]

            ssm, z, out_t = self.decode_one_token(q_t, k_t, v_t, ssm, z)
            output.append(out_t)

        output = torch.stack(output, dim=2)
        output = self.merge_heads(output)
        return output @ self.o

    def decode_one_token(self, q_t, k_t, v_t, ssm, z):
        ssm = ssm + k_t[..., :, None] @ v_t[..., None, :]
        z = z + k_t

        numerator = q_t[..., None, :] @ ssm
        denominator = q_t[..., None, :] @ z[..., :, None]
        out_t = numerator.squeeze(-2) / denominator.squeeze(-1).clamp_min(1e-6)

        return ssm, z, out_t


def main():
    torch.manual_seed(0)

    batch_size = 2
    seq_len = 4
    hidden_dim = 8
    num_heads = 2

    q = torch.randn(batch_size, seq_len, hidden_dim)
    k = torch.randn(batch_size, seq_len, hidden_dim)
    v = torch.randn(batch_size, seq_len, hidden_dim)

    reference = LinearAttention(hidden_dim, batch_size, seq_len, num_heads)

    prefill_out = reference.forward_prefill(q, k, v)
    decode_out = reference.forward_decode(q, k, v)

    print("linear attention")
    print("prefill shape:", tuple(prefill_out.shape))
    print("decode shape: ", tuple(decode_out.shape))
    print("max abs diff:", (prefill_out - decode_out).abs().max().item())
    print("close:       ", torch.allclose(prefill_out, decode_out, atol=1e-6))
    print()


if __name__ == "__main__":
    main()
