# model-design-for-faster-inference-attention

learning different model design patterns for faster inference

01-single-head-attention.py - a regular single head attention

what do you think the memory cost is to save kv cache for a single headed attention?

cost = batch_size * seq_len * num_attn_layers * 2(one for k and one for v) * hidden_dim * dtype

02-multi-head-attention.py - single head attention extended to multi head attention

lets calculate some memory cost to save the kv cache here when we use MHA -

cost = batch_size * seq_len * num_attn_layers * 2 * num_kv_heads * head_dim * dtype

03-grouped-query-attention.py - a memory saving technique where unlike MHA, multiple query heads share the same key/value heads.

in standard MHA, each query head has its own key and value head. Q = K = V
in GQA, the query heads are more than the KV heads and they attend to a shared KV head.

for example -
if you num query heads = 8 and a group size of 2, 

total kv heads = 8 / 2 = 4

Q[0, 1] share KV head 0
Q[2, 3] share KV head 1
Q[4, 5] share KV head 2
Q[6, 7] share KV head 3

![](./assets/gqa.png)

cost = batch_size * seq_len * num_attn_layers * 2 * num_kv_heads * head_dim * dtype

this means GQA uses less than 1/group_size of the kv cache compared to MHA.

for instance if group_size = 4, then GQA uses only 1/4 of the memory compared to MHA.

03-fa1.py - block q, k and v with online softmax

04-linear-attention.py - this needs a bit of an explanation lol.


![](./assets/linear_attention_1.png)

![](./assets/linear_attention_2.png)

![](./assets/linear_attention_prefill.png)

for the decode step, unlike full attention, the computed state in linear attention is superimposed on the previous historical intermediate matrix. this intermediate matrix is also called an SSM. Each new SSM can be directly added to all previous SSMs.

the major difference is full attention scales with L(seq len), but in linear attention it stays fixed at d x d(hidden dim).

do you see a problem here? because at each step, the attention just keeps getting added on top, you dont really know the state at each particular step is.
no key points, no important information, no discard anything, everything is there just in a big bowl.

this is probably the reason why the effective of linear attention is lesser compared to full attention.

![](./assets/linear_attention_problem.png)
