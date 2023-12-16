import argparse
import os.path as osp
from typing import Any, Dict, Optional

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch_geometric.transforms as T

from torch_geometric.data import Data

import scipy.sparse.csgraph as csgraph
import numpy as np

class RemoveFeatures(T.BaseTransform):
    def forward(self, data: Data) -> Data:
        # only keep edge_index, spd, rwpe
        data = Data(edge_index=data.edge_index, spd=data.spd, rwpe=data.rwpe, num_nodes=data.num_nodes)
        return data


class AddShortestPathMatrix(T.BaseTransform):
    
    def forward(self, data: Data) -> Data:

        data = self.add_shortest_path_matrix(data)
        
        return data
    
    def add_shortest_path_matrix(self, data: Data):
        edge_index = data.edge_index.cpu().numpy()
        num_nodes = data.num_nodes
        # adj
        adjacency_matrix = np.zeros((num_nodes, num_nodes)) # type: ignore
        adjacency_matrix[edge_index[0], edge_index[1]] = 1
        # spd
        shortest_path_matrix = csgraph.floyd_warshall(adjacency_matrix)
        # diameter
        diameter = np.max(shortest_path_matrix[shortest_path_matrix != np.inf])
        assert diameter > 0
        # normalize
        # shortest_path_matrix = shortest_path_matrix / diameter
        # save to data
        data.spd = torch.tensor(shortest_path_matrix.reshape(-1, 1), dtype=torch.float).to(data.edge_index.device)
        data.diameter = torch.tensor(diameter, dtype=torch.int).to(data.edge_index.device)
        return data

# transform = AddShortestPathMatrix()
