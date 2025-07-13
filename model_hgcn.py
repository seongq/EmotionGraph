import torch
import torch.nn as nn
from itertools import permutations
import geoopt
from geoopt.manifolds.lorentz import Lorentz
from hgcn import HGCNBlock  # assumes your earlier defined HGCNBlock with softplus kappa
import random


class HGCN(nn.Module):
    def __init__(self, alpha, variant, lamda, a_dim, v_dim, l_dim, n_dim,nlayers, nhidden, nclass, dropout, return_feature, use_residue,
                 new_graph='full', n_speakers=2, modals=['a','v','l'], use_speaker=True, use_modal=False, num_K=4, kappa_learnable=True, kappa=1.0, edge_type="M3NET"):
        super(HGCN, self).__init__()
        self.kappa_learnable = kappa_learnable
        self.kappa = kappa
        self.return_feature = return_feature
        self.use_residue = use_residue
        self.new_graph = new_graph
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.modals = modals
        self.modal_embeddings = nn.Embedding(3, n_dim)
        self.speaker_embeddings = nn.Embedding(n_speakers, n_dim)
        self.use_speaker = use_speaker
        self.use_modal = use_modal
        self.use_position = False

        self.fc1 = nn.Linear(n_dim, nhidden)
        self.num_K = num_K

        self.edge_type = edge_type

        for kk in range(num_K):
            setattr(self, f'hgcn{kk+1}', HGCNBlock(dim=nhidden, learnable_kappa=self.kappa_learnable , dropout=dropout, kappa=self.kappa))

    def forward(self, a, v, l, dia_len, qmask, epoch):
        qmask = torch.cat([qmask[:x, i, :] for i, x in enumerate(dia_len)], dim=0)
        spk_idx = torch.argmax(qmask, dim=-1)
        spk_emb_vector = self.speaker_embeddings(spk_idx)

        if self.use_speaker and 'l' in self.modals:
            l += spk_emb_vector

        if self.use_modal:
            emb_idx = torch.LongTensor([0, 1, 2]).to(a.device)
            emb_vector = self.modal_embeddings(emb_idx)
            if 'a' in self.modals:
                a += emb_vector[0].unsqueeze(0).expand(a.shape[0], -1)
            if 'v' in self.modals:
                v += emb_vector[1].unsqueeze(0).expand(v.shape[0], -1)
            if 'l' in self.modals:
                l += emb_vector[2].unsqueeze(0).expand(l.shape[0], -1)

        edge_index, features = self.create_gnn_index(a, v, l, dia_len, self.modals)
        # if self.training:
        #     # print("training")
        #     if random.choice([True,False]):
        #         edge_index = self.create_random_tree_index(edge_index, dia_len, self.modals)
        #     else:
        #         edge_index = edge_index
        #     x = self.fc1(features)
        # else:
        #    # print("training")
        #     if random.choice([True,False]):
        #         edge_index = self.create_random_tree_index(edge_index, dia_len, self.modals)
        #     else:
        #         edge_index = edge_index
        x = self.fc1(features)
        for kk in range(self.num_K):
            x = getattr(self, f'hgcn{kk+1}')(x, edge_index)

        # x = x_h

        if self.use_residue:
            x = torch.cat([features, x], dim=-1)

        

        x = self.reverse_features(dia_len, x)
        # print(x)
        return x

    def create_random_tree_index(self,edge_index: torch.Tensor, dia_len, modals):
        """
        edge_index: [2, E] 형태의 연결 정보 (disjoint graph)
        dia_len: 각 dialogue의 길이 리스트
        modals: 예: ['a', 'v', 'l']
        """
        num_modality = len(modals)
        device = edge_index.device
        total_tree_edges = []

        node_ptr = 0
        for length in dia_len:
            # 해당 dialogue의 노드 범위
            num_nodes = length * num_modality
            nodes = list(range(node_ptr, node_ptr + num_nodes))

            # 원래 edge_index에서 해당 subgraph에 속한 edge만 추출
            mask = ((edge_index[0] >= node_ptr) & (edge_index[0] < node_ptr + num_nodes))
            local_edges = edge_index[:, mask]
            
            # 로컬 노드 번호로 변환
            local_edges = local_edges - node_ptr

            # 무향 edge 집합 만들기
            edge_set = set()
            for i in range(local_edges.size(1)):
                u, v = local_edges[0, i].item(), local_edges[1, i].item()
                if u != v:
                    edge_set.add(tuple(sorted((u, v))))
            edge_list = list(edge_set)

            # Kruskal-like 방식으로 spanning tree 만들기
            parent = list(range(num_nodes))

            def find(u):
                while parent[u] != u:
                    parent[u] = parent[parent[u]]
                    u = parent[u]
                return u

            def union(u, v):
                parent[find(u)] = find(v)

            random.shuffle(edge_list)
            count = 0
            tree_edges = []
            for u, v in edge_list:
                if find(u) != find(v):
                    union(u, v)
                    # 다시 글로벌 노드 번호로 변환
                    tree_edges.append((u + node_ptr, v + node_ptr))
                    count += 1
                    if count == num_nodes - 1:
                        break

            total_tree_edges.extend(tree_edges)
            node_ptr += num_nodes

        final_tree_index = torch.LongTensor(total_tree_edges).T.to(device)
        return final_tree_index

    def create_gnn_index(self, a, v, l, dia_len, modals):
        # print(sum(dia_len))
        self_loop = False
        num_modality = len(modals)
        node_count = 0
        batch_count = 0
        index =[]
        tmp = []
        
        for i in dia_len:
            nodes = list(range(i*num_modality))
            nodes = [j + node_count for j in nodes]
            nodes_l = nodes[0:i*num_modality//3]
            nodes_a = nodes[i*num_modality//3:i*num_modality*2//3]
            nodes_v = nodes[i*num_modality*2//3:]
            index = index + list(permutations(nodes_l,2)) + list(permutations(nodes_a,2)) + list(permutations(nodes_v,2))
            Gnodes=[]
            for _ in range(i):
                Gnodes.append([nodes_l[_]] + [nodes_a[_]] + [nodes_v[_]])
            for ii, _ in enumerate(Gnodes):
                tmp = tmp +  list(permutations(_,2))
            if node_count == 0:
                ll = l[0:0+i]
                aa = a[0:0+i]
                vv = v[0:0+i]
                features = torch.cat([ll,aa,vv],dim=0)
                temp = 0+i
            else:
                ll = l[temp:temp+i]
                aa = a[temp:temp+i]
                vv = v[temp:temp+i]
                features_temp = torch.cat([ll,aa,vv],dim=0)
                features =  torch.cat([features,features_temp],dim=0)
                temp = temp+i
            node_count = node_count + i*num_modality
        edge_index = torch.cat([torch.LongTensor(index).T,torch.LongTensor(tmp).T],1).cuda()

        return edge_index, features

    def reverse_features(self, dia_len, features):
        l, a, v = [], [], []
        for i in dia_len:
            ll = features[0:1*i]
            aa = features[1*i:2*i]
            vv = features[2*i:3*i]
            features = features[3*i:]
            l.append(ll)
            a.append(aa)
            v.append(vv)
        tmpl = torch.cat(l, dim=0)
        tmpa = torch.cat(a, dim=0)
        tmpv = torch.cat(v, dim=0)
        features = torch.cat([tmpl, tmpa, tmpv], dim=-1)
        return features
