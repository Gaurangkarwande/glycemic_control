import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN
from torch_geometric.nn.pool import global_mean_pool

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features: int, batch_size: int) -> None:
        """Initialize model

        Args:
            node_features: the input temporal sequence lenght
            batch_size: the batch_size
        """
        super(RecurrentGCN, self).__init__()
        self.recurrent = DCRNN(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, 1)
        self.batch_size = batch_size

    def forward(
        self,
        x: torch.tensor,
        edge_index: torch.tensor,
        edge_weight: torch.tensor,
        batch: torch.tensor,
    ) -> torch.tensor:
        """The forward method

        Args:
            x: the feature tensor. Shape = (batch_size * num_nodes) x node_features
            edge_index: the COO adjacency matrix. Shape = 2 x (num_nodes C 2 * 2 * batch_size)
            edge_weight: the matirx defining edge weights. Shape = (num_nodes C 2 * 2 * batch_size)
            batch: the tensor specifying which node belongs to which graph in a batch.
                Shape = (batch_size * num_nodes)
        """
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        h = global_mean_pool(h, size=self.batch_size, batch=batch)
        return h
