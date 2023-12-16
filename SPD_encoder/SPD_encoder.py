from itertools import count
from re import M
import torch
from torch_geometric.utils import to_networkx
from torch_geometric.utils.num_nodes import maybe_num_nodes
import torch_geometric.nn as pyg_nn
import torch.nn.init as init

import networkx as nx


class SPDEncoder(torch.nn.Module):
    def __init__(self, embedding_dim=64):
        super(SPDEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        
        self.GPS = GPS(channels=self.embedding_dim, pe_dim=20, num_layers=10, attn_type='performer',
            attn_kwargs={'dropout': 0.5})

        self.sim_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * self.embedding_dim, self.embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_dim, self.embedding_dim//2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_dim//2, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x, edge_index, batch):

        x = self.GPS(x, edge_index, batch)

        return x
    
    # def similarity(self, z1, z2, gamma=0.015625):
    #     # MLP 
    #     # Radial basis function kernel
    #     # gamma = 1 / (2 * self.embedding_dim)
    #     return torch.exp(-gamma * (z1 - z2).norm(dim=-1))

    # def loss(self, z1 ,z2, n1, n2, device='cpu'):
    #     # Compute SPD similarity
    #     spd_sim = 1 / self.spd_matrix[n1, n2]
    #     spd_sim = spd_sim.to(device)
    #     # Compute embedding similarity
    #     z_sim = self.similarity(z1, z2)
    #     # Compute loss
    #     loss = torch.nn.MSELoss()(spd_sim, z_sim)
        
        # return loss
    def batch_loss(self, out, batch_spd, batch_ptr, batch_size, device='cpu', max_pairs_per_graph=6400):
        total_loss = 0.0
        start_idx = 0
        count_pairs = 0
        for i in range(batch_size):
            # Extract the SPD matrix for the current graph in the batch
            num_nodes = batch_ptr[i + 1] - batch_ptr[i]
            end_idx = start_idx + num_nodes ** 2
            spd_matrix = batch_spd[start_idx: end_idx].view(num_nodes, num_nodes)
            spd_matrix = self.spd_to_sim(spd_matrix)
            # Extract the embeddings for the current graph in the batch
            embeddings = out[batch_ptr[i]:batch_ptr[i+1]]
            # Select node pairs
            triu_indices = torch.triu_indices(row=num_nodes, col=num_nodes, offset=1)
            if max_pairs_per_graph < 0:
                random_indices = torch.randperm(triu_indices.size(1))[:max_pairs_per_graph]
                selected_indices = triu_indices[:, random_indices]



                # Compute the similarity matrix for the embeddings
                z_sim_matrix = self.node_pairs_similarity(embeddings, device=device, selected_indices=selected_indices)
                spd_matrix = spd_matrix[selected_indices[0], selected_indices[1]]
            else:
                max_pairs_per_graph = min(max_pairs_per_graph, triu_indices.size(1))
                # 层次抽样
                input_tensor = spd_matrix[triu_indices[0], triu_indices[1]]

                # torch.save(spd_matrix, f'spd_matrix{i}.pt')
                # 参数设置
                num_intervals = 10
                samples_per_interval = max_pairs_per_graph // num_intervals

                # 预先计算所有区间的索引
                interval_indices = [(input_tensor >= i/num_intervals) & (input_tensor < (i+1)/num_intervals) for i in range(num_intervals)]

                # for i in range(num_intervals):
                #     print(i, interval_indices[i].sum())

                # 优化的抽样过程，允许重复抽样
                sampled_indices = torch.cat([
                    torch.multinomial(interval_indices[i].float(), samples_per_interval, replacement=True)
                    for i in range(num_intervals) if interval_indices[i].sum() > 0  # 确保区间内至少有一个元素
                ])
                selected_indices = triu_indices[:, sampled_indices]
                z_sim_matrix = self.node_pairs_similarity(embeddings, device=device, selected_indices=selected_indices)
                spd_matrix = spd_matrix[selected_indices[0], selected_indices[1]]
                

            # Compute the SPD similarity matrix
            # spd_sim_matrix = 1 / (spd_matrix + 1e-15)
            # lambda_ = torch.tensor(3)
            # spd_sim_matrix = (torch.exp(- lambda_ * spd_matrix) - torch.exp(- lambda_)) * ( 1/(1 - torch.exp(- lambda_)))
            # spd_sim_matrix = spd_sim_matrix.to(device)

            spd_sim_matrix = spd_matrix.to(device)

            # Calculate the loss for the current graph
            # mse_loss = torch.nn.functional.mse_loss(z_sim_matrix, spd_sim_matrix, reduction='mean')

            # mse_loss = torch.nn.functional.mse_loss(z_sim_matrix[selected_indices[0], selected_indices[1]], spd_sim_matrix[selected_indices[0], selected_indices[1]], reduction='mean')
            mse_loss = torch.nn.functional.mse_loss(z_sim_matrix, spd_sim_matrix, reduction='sum')
            count_pairs += len(selected_indices[0])

            total_loss += mse_loss

            start_idx = end_idx
        
        # Average loss by the number of graphs in the batch
        # return total_loss / batch_size
        return total_loss, count_pairs

    def node_pairs_similarity(self, embeddings, device='cpu', selected_indices=None, sim_mlp=True):
        if sim_mlp==True and selected_indices != None:
            z1 = embeddings[selected_indices[0]]  # Shape (num_pairs, embedding_dim)
            z2 = embeddings[selected_indices[1]]  # Shape (num_pairs, embedding_dim)

            # Concatenate embeddings for each pair
            # z_pairs = torch.cat((z1, z2), dim=-1)  # Shape (num_pairs, 2 * embedding_dim)
            # # Apply sim_mlp to concatenated embeddings
            # similarity_vector = self.sim_mlp(z_pairs).squeeze()

            z_pairs_forward = torch.cat((z1, z2), dim=-1)  # Shape (num_pairs, 2 * embedding_dim)
            z_pairs_backward = torch.cat((z2, z1), dim=-1)  # Shape (num_pairs, 2 * embedding_dim)

            # Apply sim_mlp to concatenated embeddings in both directions
            similarity_vector_forward = self.sim_mlp(z_pairs_forward).squeeze()
            similarity_vector_backward = self.sim_mlp(z_pairs_backward).squeeze()

            # Combine the results from both directions
            # You can use different strategies like mean, max, or a learned combination
            similarity_vector = (similarity_vector_forward + similarity_vector_backward) / 2

            return similarity_vector.to(device)


        if selected_indices == None:
            # Calculate pairwise distance matrix for embeddings
            z1 = embeddings.unsqueeze(1)  # Shape (num_nodes, 1, embedding_dim)
            z2 = embeddings.unsqueeze(0)  # Shape (1, num_nodes, embedding_dim)
            distance_matrix = torch.norm(z1 - z2, dim=-1, p=2)  # Shape (num_nodes, num_nodes)
            
            # Apply the radial basis function (RBF) kernel to the distance matrix
            gamma = 0.015625  # You can adjust this value as needed
            rbf_kernel_matrix = torch.exp(-gamma * distance_matrix.pow(2))
            return rbf_kernel_matrix.to(device)
        else:
            # Calculate pairwise distance matrix for embeddings
            z1 = embeddings[selected_indices[0]]  # Shape (num_pairs, embedding_dim)
            z2 = embeddings[selected_indices[1]]  # Shape (num_pairs, embedding_dim)

            # Calculate pairwise distances only for selected pairs
            distance_vector = torch.norm(z1 - z2, dim=-1, p=2)  # Shape (num_pairs,)

            # Apply the radial basis function (RBF) kernel to the distance vector
            gamma = 0.015625  # This value can be adjusted as needed
            # rbf_kernel_vector = torch.exp(-gamma * distance_vector.pow(2))
            rbf_kernel_vector = torch.exp(-gamma * distance_vector.pow(2))


            return rbf_kernel_vector.to(device)
        
    def spd_to_sim(self, spd_vector,  alpha=1.152):
        # return np.power(alpha, 1-x) 
        return torch.pow(torch.tensor(alpha), 1-spd_vector)


import argparse
import os.path as osp
from typing import Any, Dict, Optional

import torch
from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch_geometric.transforms as T
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, GPSConv, global_add_pool, GINConv
from torch_geometric.nn.attention import PerformerAttention

class GPS(torch.nn.Module):
    def __init__(self, channels: int, pe_dim: int, num_layers: int,
                 attn_type: str, attn_kwargs: Dict[str, Any]):
        super().__init__()

        # Removed self.node_emb
        self.pe_lin = Linear(pe_dim, channels)  # Adjust dimensions according to your Laplacian PE
        self.pe_norm = BatchNorm1d(pe_dim)  # Adjust dimensions according to your Laplacian PE

        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            ) 
            conv = GPSConv(channels, GINConv(nn), heads=4,
                           attn_type=attn_type, attn_kwargs=attn_kwargs)
            self.convs.append(conv)

        # self.mlp = Sequential(
        #     Linear(channels, channels // 2),
        #     ReLU(),
        #     Linear(channels // 2, channels // 4),
        #     ReLU(),
        #     Linear(channels // 4, 1),
        # )

        self.mlp = Sequential(
            Linear(channels, channels),
            ReLU(),
            Linear(channels, channels*2),
            ReLU(),
            Linear(channels*2, channels),
        )
        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if attn_type == 'performer' else None)

    def forward(self, pe, edge_index, batch):
        # x_pe = self.pe_norm(pe)  # Assume laplacian_pe is calculated externally
        # x = self.pe_lin(x_pe)  # Now x contains only the Laplacian PE

        x = self.pe_lin(pe)
                        
        # edge_attr = self.edge_emb(edge_attr)

        for conv in self.convs:
            x = conv(x, edge_index, batch)

        out = self.mlp(x)
        return out


class RedrawProjection:
    def __init__(self, model: torch.nn.Module,
                 redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# attn_kwargs = {'dropout': 0.5}
# model = GPS(channels=64, pe_dim=20, num_layers=10, attn_type=args.attn_type,
#             attn_kwargs=attn_kwargs)