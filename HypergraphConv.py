from typing import Optional

import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import softmax
from torch_geometric.nn.conv import MessagePassing
#from torch_geometric.nn import aggr
from torch_geometric.nn.inits import glorot, zeros
from torch import nn
from torch_scatter import scatter

def com_mult(a, b):
	r1, i1 = a[..., 0], a[..., 1]
	r2, i2 = b[..., 0], b[..., 1]
	return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim = -1)

def conj(a):
	a[..., 1] = -a[..., 1]
	return a

def ccorr(a, b):
	return torch.irfft(com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)
class HypergraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, max_nodes=5000, max_edges=5000, dropout=0.5):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.dropout = dropout

        # 학습 가능한 weight들
        self.edge_weights = nn.Parameter(torch.ones(max_edges))            # (E,)
        self.H_hat_full = nn.Parameter(torch.randn(max_nodes, max_edges))  # (N, E)
        self.linear = nn.Linear(in_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x, edge_index):
        """
        x: (N, in_dim)         -- node features
        edge_index: (2, E)     -- (src, tgt), each ∈ [0, N-1]
        """
        N = x.size(0)
        E = edge_index.size(1)
        device = x.device

        # Step 1: binary incidence matrix 만들기
        H_bin = torch.zeros(N, E, device=device)
        src, tgt = edge_index  # (2, E)
        H_bin[src, torch.arange(E, device=device)] = 1.0
        H_bin[tgt, torch.arange(E, device=device)] = 1.0

        # Step 2: 학습 weight들 준비
        H_hat = self.H_hat_full[:N, :E]                # (N, E)
        edge_w = F.relu(self.edge_weights[:E])         # (E,)
        W = torch.diag(edge_w)                         # (E, E)
        H_weighted = H_bin * F.relu(H_hat)             # (N, E)

        # Step 3: degree matrices
        DH = torch.diag(torch.sum(H_weighted @ W, dim=1) + 1e-6)  # (N, N)
        B  = torch.diag(torch.sum(H_weighted, dim=0) + 1e-6)      # (E, E)
        DH_inv = torch.inverse(DH)
        B_inv  = torch.inverse(B)

        # Step 4: 선형 변환
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear(x)  # (N, out_dim)

        # Step 5: HGCN propagation
        out = DH_inv @ H_weighted @ W @ B_inv @ H_weighted.T @ x  # (N, out_dim)
        return F.relu(out)
# class HypergraphConv(MessagePassing):
#     def __init__(self, in_channels, out_channels, use_attention=False, heads=1,
#                  concat=True, negative_slope=0.2, dropout=0.0, bias=True):
#         super().__init__(aggr='add', node_dim=0, flow="source_to_target")
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.use_attention = use_attention
#         self.heads = heads if use_attention else 1
#         self.concat = concat if use_attention else True
#         self.negative_slope = negative_slope
#         self.dropout = dropout

#         self.weight = Parameter(torch.Tensor(in_channels, self.heads * out_channels))
#         if use_attention:
#             self.att = Parameter(torch.Tensor(1, self.heads, 2 * out_channels))

#         if bias:
#             self.bias = Parameter(torch.Tensor(self.heads * out_channels if self.concat else out_channels))
#         else:
#             self.register_parameter('bias', None)

#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.weight)
#         if self.use_attention:
#             nn.init.xavier_uniform_(self.att)
#         if self.bias is not None:
#             nn.init.zeros_(self.bias)

#     def forward(self, x: Tensor, hyperedge_index: Tensor,
#                 hyperedge_weight: Tensor = None,
#                 hyperedge_attr: Tensor = None) -> Tensor:
#         num_nodes = x.size(0)
#         num_edges = int(hyperedge_index[1].max()) + 1 if hyperedge_index.numel() > 0 else 0

#         x = torch.matmul(x, self.weight).view(-1, self.heads, self.out_channels)

#         if hyperedge_weight is None:
#             hyperedge_weight = x.new_ones(num_edges)

#         alpha = None
#         if self.use_attention:
#             assert hyperedge_attr is not None
#             hyperedge_attr = torch.matmul(hyperedge_attr, self.weight).view(-1, self.heads, self.out_channels)
#             x_i = x[hyperedge_index[0]]
#             x_j = hyperedge_attr[hyperedge_index[1]]
#             alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
#             alpha = F.leaky_relu(alpha, self.negative_slope)
#             alpha = softmax(alpha, hyperedge_index[0], num_nodes=num_nodes)
#             alpha = F.dropout(alpha, p=self.dropout, training=self.training)
#         print("num odes", num_nodes)
#         print("num edges", num_edges)
#         print("hyperedge[0] size", len(hyperedge_index[0]))
#         print("hyperedge[1] size", len(hyperedge_index[1]))
#         # Node → Hyperedge
#         deg = scatter(hyperedge_weight[hyperedge_index[1]], hyperedge_index[0], dim=0,
#                       dim_size=num_nodes, reduce='sum')
#         norm = 1.0 / deg.clamp(min=1e-10)
#         self.flow = "source_to_target"
#         out = self.propagate(hyperedge_index, x=x, norm=norm, alpha=alpha, size=(num_nodes, num_edges))

#         # Hyperedge → Node
#         deg = scatter(torch.ones_like(hyperedge_index[0]), hyperedge_index[1], dim=0,
#                       dim_size=num_edges, reduce='sum')
#         norm = 1.0 / deg.clamp(min=1e-10)
#         self.flow = "target_to_source"
#         out = self.propagate(hyperedge_index, x=out, norm=norm, alpha=alpha, size=(num_edges, num_nodes))


#         if self.concat and out.dim() == 3:
#             out = out.view(-1, self.heads * self.out_channels)
#         elif not self.concat:
#             out = out.mean(dim=1)

#         if self.bias is not None:
#             out = out + self.bias

#         return F.leaky_relu(out)

#     def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor = None) -> Tensor:
#         out = norm_i.view(-1, 1, 1) * x_j
#         if alpha is not None:
#             out = alpha.view(-1, self.heads, 1) * out
#         return out

#     def __repr__(self):
#         return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels})'

# class HypergraphConv(MessagePassing):
#     """Args:
#         in_channels (int): Size of each input sample.
#         out_channels (int): Size of each output sample.
#         use_attention (bool, optional): If set to :obj:`True`, attention
#             will be added to this layer. (default: :obj:`False`)
#         heads (int, optional): Number of multi-head-attentions.
#             (default: :obj:`1`)
#         concat (bool, optional): If set to :obj:`False`, the multi-head
#             attentions are averaged instead of concatenated.
#             (default: :obj:`True`)
#         negative_slope (float, optional): LeakyReLU angle of the negative
#             slope. (default: :obj:`0.2`)
#         dropout (float, optional): Dropout probability of the normalized
#             attention coefficients which exposes each node to a stochastically
#             sampled neighborhood during training. (default: :obj:`0`)
#         bias (bool, optional): If set to :obj:`False`, the layer will not learn
#             an additive bias. (default: :obj:`True`)
#         **kwargs (optional): Additional arguments of
#             :class:`torch_geometric.nn.conv.MessagePassing`.
#     """
#     def __init__(self, in_channels, out_channels, use_attention=False, heads=1,
#                  concat=True, negative_slope=0.2, dropout=0, bias=True,
#                  **kwargs):
#         kwargs.setdefault('aggr', 'add')
#         super(HypergraphConv, self).__init__(node_dim=0,  **kwargs)

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.use_attention = use_attention
#         if self.use_attention:
#             self.heads = heads
#             self.concat = concat
#             self.negative_slope = negative_slope
#             self.dropout = dropout
#             self.weight = Parameter(
#                 torch.Tensor(in_channels, heads * out_channels))
#             self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))
#         else:
#             self.heads = 1
#             self.concat = True
#             self.weight = Parameter(torch.Tensor(in_channels, out_channels))
#         self.edgeweight = Parameter(torch.Tensor(in_channels, out_channels))
#         if bias and concat:
#             self.bias = Parameter(torch.Tensor(heads * out_channels))
#         elif bias and not concat:
#             self.bias = Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)
#         self.edgefc = torch.nn.Linear(in_channels, out_channels)
#         self.reset_parameters()

#     def reset_parameters(self):
#         glorot(self.weight)
#         glorot(self.edgeweight)
#         if self.use_attention:
#             glorot(self.att)
#         zeros(self.bias)


#     def forward(self, x, hyperedge_index,
#                 hyperedge_weight = None,
#                 hyperedge_attr = None, EW_weight = None, dia_len = None):
#         """
#         Args:
#             x (Tensor): Node feature matrix
#                 :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
#             hyperedge_index (LongTensor): The hyperedge indices, *i.e.*
#                 the sparse incidence matrix
#                 :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` mapping from
#                 nodes to edges.
#             hyperedge_weight (Tensor, optional): Hyperedge weights
#                 :math:`\mathbf{W} \in \mathbb{R}^M`. (default: :obj:`None`)
#             hyperedge_attr (Tensor, optional): Hyperedge feature matrix in
#                 :math:`\mathbb{R}^{M \times F}`.
#                 These features only need to get passed in case
#                 :obj:`use_attention=True`. (default: :obj:`None`)
#         """
#         num_nodes, num_edges = x.size(0), 0
#         if hyperedge_index.numel() > 0:
#             num_edges = int(hyperedge_index[1].max()) + 1
#         if hyperedge_weight is None:
#             hyperedge_weight = x.new_ones(num_edges)
#         #if hyperedge_attr is not None:
#         #    hyperedge_attr = self.edgefc(hyperedge_attr)
#         #x = torch.matmul(x, self.weight)
#         alpha = None
#         if self.use_attention:
#             assert hyperedge_attr is not None
#             x = x.view(-1, self.heads, self.out_channels)
#             hyperedge_attr = torch.matmul(hyperedge_attr, self.weight)
#             hyperedge_attr = hyperedge_attr.view(-1, self.heads,
#                                                  self.out_channels)
#             x_i = x[hyperedge_index[0]]
#             x_j = hyperedge_attr[hyperedge_index[1]]
#             alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
#             alpha = F.leaky_relu(alpha, self.negative_slope)
#             alpha = softmax(alpha, hyperedge_index[0], num_nodes=x.size(0))
#             alpha = F.dropout(alpha, p=self.dropout, training=self.training)

#         D = scatter_add(hyperedge_weight[hyperedge_index[1]],
#                         hyperedge_index[0], dim=0, dim_size=num_nodes)      #[num_nodes]
#         D = 1.0 / D                                                         # all 0.5 if hyperedge_weight is None
#         D[D == float("inf")] = 0
#         if EW_weight is None:
#             B = scatter_add(x.new_ones(hyperedge_index.size(1)),
#                         hyperedge_index[1], dim=0, dim_size=num_edges)      #[num_edges]
#         else:
#             B = scatter_add(EW_weight[hyperedge_index[0]],
#                         hyperedge_index[1], dim=0, dim_size=num_edges)      #[num_edges]
#         B = 1.0 / B
#         B[B == float("inf")] = 0
#         self.flow = 'source_to_target'
#         print(hyperedge_index[0][0])
#         out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha,#hyperedge_attr[hyperedge_index[1]],  
#                              size=(num_nodes, num_edges))                   #num_edges,1,100
#         self.flow = 'target_to_source'
        
#         out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha,
#                              size=(num_edges, num_nodes))                   #num_nodes,1,100
#         if self.concat is True and out.size(1) == 1:
#             out = out.view(-1, self.heads * self.out_channels)
#         else:
#             out = out.mean(dim=1)

#         if self.bias is not None:
#             out = out + self.bias
        
#         return torch.nn.LeakyReLU()(out)  #


#     def message(self, x_j, norm_i, alpha):
#         H, F = self.heads, self.out_channels
#         out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)
#         if alpha is not None:
#             out = alpha.view(-1, H, 1) * out
#         return out

    # def message(self, x_j, norm_i, alpha):
    #     H, F = self.heads, self.out_channels

    #     if self.flow == 'source_to_target':
    #         # Node → Hyperedge
    #         out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)
    #         if alpha is not None:
    #             out = alpha.view(-1, H, 1) * out
    #         return out

    #     elif self.flow == 'target_to_source':
    #         # Hyperedge → Node
    #         out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)
    #         if alpha is not None:
    #             out = alpha.view(-1, H, 1) * out
    #         return out
    # def message(self, x_j, norm_i, alpha):
    #     H, F = self.heads, self.out_channels

    #     if x_j.dim() == 2:
    #         out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)
    #     else:
    #         out = norm_i.view(-1, 1, 1) * x_j      
    #     if alpha is not None:
    #         out = alpha.view(-1, self.heads, 1) * out

    #     return out

    #def update(self, aggr_out):
    #    return aggr_out

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
