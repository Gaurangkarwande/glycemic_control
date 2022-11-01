import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN
from torch_geometric.nn.pool import global_mean_pool
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentGCN, self).__init__()
        self.recurrent = DCRNN(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, 1)
        self.batch = torch.as_tensor(np.zeros(16), dtype=torch.int64).to(device)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        h = global_mean_pool(h, size=1, batch=self.batch)
        return h
