from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from .GraphLayers import GNN
from torch_geometric.data import data, batch


class GraphClassification(nn.Module):
    def __init__(self, config: Dict[str, Any], num_nodes: int, pretraining: bool = True) -> None:
        super().__init__()
        self.lstm_layers = nn.ModuleList(
            [
                nn.LSTM(
                    input_size=config["lstm_input_dim"],
                    hidden_size=config["lstm_hidden_dim"],
                    num_layers=config["lstm_num_layers"],
                    dropout=config["dropout"],
                    bidirectional=True,
                    batch_first=True,
                )
                for _ in range(num_nodes)
            ]
        )

        self.gnn = GNN(config=config, num_nodes=num_nodes, pretraining=pretraining)

        self.fc = nn.Sequential(
            nn.Dropout(p=config["dropout"]),
            nn.ReLU(),
            nn.Linear(in_features=num_nodes, out_features=config["num_classes"]),
        )

        self.num_nodes = num_nodes
        self.pretraining = pretraining

    def forward(
        self,
        data: Union[data.Data, batch.Batch],
        device: Any,
        batch_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """The forward method

        Args:

        """

        # get hidden encoding of time varying variables
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        seq_lens = data.seq_lens
        if batch_size is None:
            batch_size = data.num_graphs

        if batch_size > 1:
            x = x.view(batch_size, -1, self.num_nodes)  # batch_size x seq_len x num_nodes

        gnn_input = []

        # use pack padded sequence before feeding to lstm layer
        for i, lstm_layer in enumerate(self.lstm_layers):
            if self.pretraining:
                lstm_input = nn.utils.rnn.pack_padded_sequence(
                    x[:, :, i].unsqueeze(-1), seq_lens, batch_first=True, enforce_sorted=False
                )
            else:
                lstm_input = x[i, :seq_lens].unsqueeze(1)
                lstm_input = lstm_input.unsqueeze(0)

            # print(lstm_input)
            _, (hn, cn) = lstm_layer(
                lstm_input
            )  # hn: num_lstm_layers*2 x batch_size x lstm_hidden_dim
            hn = hn.view(batch_size, -1)  # batch_size x num_lstm_layers * 2 * lstm_hidden_dim
            gnn_input.append(hn)

        # and then pass to gat
        gnn_input = torch.stack(
            gnn_input, dim=1
        )  # batch_size x num_nodes x num_lstm_layers * 2 * lstm_hidden_dim
        out, edge_index, edge_weights = self.gnn(gnn_input, batch_size, edge_index)

        # pass through fc
        out = self.fc(out)

        if self.pretraining:
            return out, None, None
        return out, edge_index, edge_weights


class LSTMClassification(nn.Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=16,
            hidden_size=config["lstm_hidden_dim"],
            num_layers=config["lstm_num_layers"],
            dropout=config["dropout"],
            bidirectional=True,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            # nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(in_features=config["lstm_hidden_dim"]*2*config["lstm_num_layers"], out_features=config["num_classes"]),
        )

    def forward(self, x, seq_lens):

        batch_size = x.shape[0]
        x = x.view(batch_size, -1, 16)
        x = nn.utils.rnn.pack_padded_sequence(x, seq_lens, batch_first=True, enforce_sorted=False)
        _, (hn, cn) = self.lstm(x)

        hn = hn.view(batch_size, -1)
        out = self.fc(hn)
        return out
