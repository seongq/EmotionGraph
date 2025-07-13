import torch
import torch.nn as nn
import geoopt
from geoopt.manifolds.lorentz import Lorentz
from utils import *
import math

import torch.nn.init as init
class HGCNBlock(nn.Module):
    def __init__(self, dim, kappa=1.0, learnable_kappa=False, dropout=0.5, act=nn.ReLU()):
        super().__init__()
        self.dim = dim
        self.dropout = dropout
        self.act = act

        if learnable_kappa:
            # raw_kappa는 unconstrained 파라미터, exp를 취하면 항상 양수
            self.raw_kappa = nn.Parameter(torch.log(torch.tensor(kappa, dtype=torch.float64)))
        else:
            self.register_buffer('raw_kappa', torch.log(torch.tensor(kappa, dtype=torch.float64)))

        # manifold는 forward에서 동적으로 생성
        self.gate = nn.Linear(2 * dim, 1, dtype=torch.float64)

        self.reset_parameters()

    def get_manifold(self):
        kappa = torch.exp(self.raw_kappa)  # ensure positive curvature
        return Lorentz(kappa)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.gate.weight, gain=math.sqrt(2))
        nn.init.zeros_(self.gate.bias)

    def forward(self, x_euc, edge_index):
        manifold = self.get_manifold()  # 곡률 업데이트 반영
        # x_H = manifold.projx(x_euc.to(torch.float64))
        x_H = manifold.expmap0(x_euc.to(torch.float64))
        x_tan = manifold.logmap0(x_H)

        N = x_tan.size(0)
        row, col = edge_index

        x_i = x_tan[col]
        x_j = x_tan[row]

        h_ij = torch.cat([x_i, x_j], dim=-1)
        alpha = torch.tanh(self.gate(h_ij))

        deg = torch.bincount(row, minlength=N).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        norm = norm.view(-1, 1)

        messages = norm * alpha * x_j

        out = torch.zeros(N, x_tan.size(1), device=x_tan.device, dtype=torch.float64)
        out.index_add_(0, col, messages)

        # x_h = manifold.projx(out)
        # x_h = manifold.expmap0(x_h)

        return out.to(torch.float32)
# class HGCNBlock(nn.Module):
#     def __init__(self, dim, kappa=1.0, learnable_kappa=False, dropout=0.5, act=nn.ReLU()):
#         super().__init__()
#         self.kappa = torch.tensor(kappa, dtype=torch.float64)
#         self.learnable_kappa = learnable_kappa
#         self.manifold = Lorentz(self.kappa, learnable_kappa)
#         self.dim = dim
#         self.gate = nn.Linear(2 * self.dim, 1,dtype=torch.float64)

        

     

#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.gate.weight, gain=math.sqrt(2))
       
#         nn.init.zeros_(self.gate.weight)


#     def forward(self, x_euc, edge_index):
#         N = x_euc.size(0)
#         # print(self.manifold.k.data)

#         # self.manifold.k.data = torch.clamp(self.manifold.k.data, min=1e-6)
#         origin = torch.zeros_like(x_euc)
#         x_H = self.manifold.projx(x_euc.to(torch.float64))
#         x_H = self.manifold.expmap0(x_H)
        
#         x_tan = self.manifold.logmap0(x_H)
        

#         N = x_tan.size(0)
#         row, col = edge_index  # source: row (j), target: col (i)
        
        
#         x_i = x_tan[col]  # [E, F] target
#         x_j = x_tan[row]  # [E, F] source

#         # Step 2: Gate attention mechanism
#         h_ij = torch.cat([x_i, x_j], dim=-1)  # [E, 2F]
#         alpha = torch.tanh(self.gate(h_ij))   # [E, 1]

#         # Step 3: Normalize
#         deg = torch.bincount(row, minlength=N).float()  # [N]
#         deg_inv_sqrt = deg.pow(-0.5)
#         deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
#         norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]  # [E]
#         norm = norm.view(-1, 1)  # [E, 1]

#         # Step 4: Weighted message
#         messages = norm * alpha * x_j  # [E, F]
        
#         # print(messages.dtype)
#         # Step 5: Aggregate
#         out = torch.zeros(N, x_tan.size(1), device=x_tan.device)  # [N, F]
#         out = out.to(torch.float64)
#         # print(out.dtype)
#         out.index_add_(0, col, messages)

        
#         # with torch.no_grad():
#         x_h = self.manifold.projx(out.to(torch.float64))
#         x_h = self.manifold.expmap0(x_h)
#         out = out.to(torch.float32)
#         x_h = x_h.to(dtype=torch.float32)
#         return out, x_h

#     # def linear_aggregate(self, x_H, edge_index, num_nodes):
#     #     edge_index = add_self_loops(edge_index, num_nodes=num_nodes)
#     #     src, dst = edge_index

#     #     x_i_edge = x_H[dst]
    #     x_j_edge = x_H[src]

    #     origin = self.manifold.origin(num_nodes, self.dim, device=x_H.device)
    #     log_x_i = self.manifold.logmap(origin[dst], x_i_edge)
    #     log_x_j = self.manifold.logmap(origin[src], x_j_edge)

    #     # log_x_i = self.rescale_if_needed(torch.nan_to_num(log_x_i, nan=0.0, posinf=0.0, neginf=0.0))
    #     # log_x_j = self.rescale_if_needed(torch.nan_to_num(log_x_j, nan=0.0, posinf=0.0, neginf=0.0))

    #     # h_pair = torch.cat([log_x_i, log_x_j], dim=1)
    #     # score = self.att_mlp(h_pair).squeeze(-1)
    #     # alpha = torch.exp(score)

    #     # norm = torch.zeros(num_nodes, dtype=torch.float64, device=x_H.device)
    #     # norm.index_add_(0, dst, alpha)
    #     # normed_alpha = alpha / (norm[dst] + 1e-8)

    #     log_msg = self.manifold.logmap(x_i_edge, x_j_edge)
    #     log_msg = self.rescale_if_needed(torch.nan_to_num(log_msg, nan=0.0, posinf=0.0, neginf=0.0))
    #     weighted = log_msg * normed_alpha.unsqueeze(1)

    #     agg_log = torch.zeros(num_nodes, x_H.size(1), dtype=torch.float64, device=x_H.device)
    #     agg_log.index_add_(0, dst, weighted)

    #     x_out = self.manifold.expmap(x_H, agg_log)
    #     return x_out


