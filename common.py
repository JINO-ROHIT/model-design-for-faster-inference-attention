import torch

def copy_weights(reference, torch_mha):
    with torch.no_grad():
        hidden_dim = reference.hidden_dim

        torch_mha.in_proj_weight.zero_()
        torch_mha.in_proj_bias.zero_()

        identity = torch.eye(hidden_dim)
        torch_mha.in_proj_weight[:hidden_dim].copy_(identity)
        torch_mha.in_proj_weight[hidden_dim:2 * hidden_dim].copy_(identity)
        torch_mha.in_proj_weight[2 * hidden_dim:].copy_(identity)

        torch_mha.out_proj.weight.copy_(reference.o.T)
        torch_mha.out_proj.bias.zero_()