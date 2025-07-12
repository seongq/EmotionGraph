import torch
import argparse

def add_self_loops(edge_index, num_nodes):
    device = edge_index.device
    loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)  # shape: (2, num_nodes)
    return torch.cat([edge_index, loop_index], dim=1)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1', 'y'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')