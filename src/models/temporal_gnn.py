import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN
from torch_geometric.nn.pool import global_mean_pool, global_max_pool


class RecurrentGCN_regression(torch.nn.Module):
    def __init__(self, node_features: int) -> None:
        """Initialize model

        Args:
            node_features: the input temporal sequence lenght
            batch_size: the batch_size
        """
        super().__init__()
        self.recurrent = DCRNN(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, 1)

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
        h = global_mean_pool(h, batch=batch)
        return h


class RecurrentGCN_classification(torch.nn.Module):
    def __init__(self, node_features: int, num_classes: int) -> None:
        """Initialize model

        Args:
            node_features: the input temporal sequence lenght
            batch_size: the batch_size
            num_classes: the number of classes
        """
        super().__init__()
        self.recurrent = DCRNN(node_features, 32, 5)
        self.linear = torch.nn.Linear(32, num_classes)

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
        h = self.recurrent(x, edge_index, edge_weight)  # batch_size * num_nodes x 32
        # print(h.shape)
        h = F.relu(h)
        h = self.linear(h)  # batch_size * num_nodes x num_classes
        h = global_mean_pool(h, batch=batch)  # batch_size x num_classes
        return h
