from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import remove_self_loops


class GNN(nn.Module):
    """The GNN module"""

    def __init__(self, config: Dict[str, Any], num_nodes: int, pretraining: bool = True) -> None:
        super().__init__()

        self.gnn_layer1 = GATv2Conv(
            in_channels=config["lstm_hidden_dim"] * config["lstm_num_layers"] * 2,
            out_channels=config["gnn_hidden_dim"],
            heads=config["gat_heads"],
        )
        self.gnn_layer2 = GATv2Conv(
            in_channels=config["gnn_hidden_dim"] * config["gat_heads"],
            out_channels=config["gnn_out_dim"],
            heads=config["gat_heads"],
            concat=False,
        )

        self.num_nodes = num_nodes
        self.pretraining = pretraining

    def forward(self, x, num_graphs, edge_index) -> Tuple[torch.Tensor, torch.Tensor]:
        if num_graphs * self.num_nodes != x.shape[0]:
            x = x.view(num_graphs * self.num_nodes, -1)

        x = self.gnn_layer1(x, edge_index)
        x, (edge_index, edge_weights) = self.gnn_layer2(
            x, edge_index, return_attention_weights=True
        )  # num_graphs * num_nodes x gnn_out_dim

        if num_graphs == 1:
            x = x.unsqueeze(0)
        else:
            x = x.view(num_graphs, self.num_nodes, -1)

        x = torch.mean(x, dim=2)
        if self.pretraining:
            return x, None, None

        edge_index, edge_weights = remove_self_loops(edge_index.detach(), edge_weights.detach())
        edge_weights = edge_weights.unsqueeze(dim=1)
        return x, edge_index, edge_weights
