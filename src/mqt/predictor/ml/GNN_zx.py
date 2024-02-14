from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn import (
    AttentionalAggregation,
    Sequential,
    TransformerConv,
    global_max_pool,
    global_mean_pool,
    models,
)


class Net(nn.Module):  # type: ignore[misc]
    num_node_categories: int = 43
    num_edge_categories: int = 2
    node_embedding_dim: int | None = None
    edge_embedding_dim: int | None = None
    num_layers: int = 1
    hidden_dim: int = 4
    output_dim: int = 7
    dropout: float = 0.0
    batch_norm: bool = False
    activation: str = "relu"
    readout: str = "mean"
    heads: int = 1
    concat: bool = False
    beta: bool = False
    bias: bool = True
    root_weight: bool = True
    model: str = "GCN"
    jk: str = "last"
    v2: bool = True

    def __init__(self, **kwargs: object) -> None:
        super().__init__()
        self.set_params(**kwargs)

        if self.node_embedding_dim and self.node_embedding_dim > 1:
            self.node_embedding = nn.Embedding(self.num_node_categories, self.node_embedding_dim)

        if self.node_embedding_dim and self.node_embedding_dim == 1:  # one-hot encoding
            self.node_embedding = lambda x: F.one_hot(x, num_classes=self.num_node_categories).float()
            self.node_embedding_dim = self.num_node_categories

        # hidden dimension accounting for multi-head concatenation
        corrected_hidden_dim = (
            (self.hidden_dim * self.heads if self.concat is True and self.heads > 1 else self.hidden_dim)
            if self.model == "TransformerConv"
            else self.hidden_dim
        )

        if self.batch_norm:
            self.batch_norm_layer = nn.BatchNorm1d(corrected_hidden_dim)
        else:
            self.batch_norm_layer = lambda x, batch: x  # noqa: ARG005

        if self.activation == "relu":
            self.activation_func = nn.ReLU()
        elif self.activation == "leaky_relu":
            self.activation_func = nn.LeakyReLU()
        elif self.activation == "tanh":
            self.activation_func = nn.Tanh()
        elif self.activation == "sigmoid":
            self.activation_func = nn.Sigmoid()

        in_dim = (self.node_embedding_dim, 1) if self.node_embedding_dim else (2,)

        if self.model == "TransformerConv":
            self.layers = []
            for i in range(self.num_layers):
                in_channels = corrected_hidden_dim if i > 0 else in_dim
                layer = Sequential(
                    "x, edge_index, edge_attr, batch",
                    [
                        (
                            TransformerConv(
                                in_channels=in_channels,
                                out_channels=self.hidden_dim,
                                heads=self.heads,
                                concat=self.concat,
                                beta=self.beta,
                                dropout=self.dropout,
                                bias=self.bias,
                                root_weight=self.root_weight,
                            ),
                            "x, edge_index, edge_attr -> x",
                        ),
                        (self.batch_norm_layer, "x, batch -> x"),
                        self.activation_func,
                    ],
                )
                self.layers.append(layer)
            last_hidden_dim = corrected_hidden_dim

        elif self.model == "GAT":
            self.layers = [
                models.GAT(
                    in_channels=-1,
                    hidden_channels=self.hidden_dim,
                    num_layers=self.num_layers,
                    out_channels=self.output_dim,
                    dropout=self.dropout,
                    act=self.activation_func,
                    norm=self.batch_norm_layer if self.batch_norm else None,
                    v2=True,
                )
            ]
            last_hidden_dim = self.output_dim

        elif self.model == "GCN":
            self.layers = [
                models.GCN(
                    in_channels=-1,
                    hidden_channels=self.hidden_dim,
                    num_layers=self.num_layers,
                    out_channels=self.output_dim,
                    dropout=self.dropout,
                    act=self.activation_func,
                    norm=self.batch_norm_layer if self.batch_norm else None,
                    edge_dim=self.edge_embedding_dim if self.edge_embedding_dim else 1,
                )
            ]
            last_hidden_dim = self.output_dim

        self.out_nn = nn.Sequential(nn.Linear(last_hidden_dim, self.output_dim), self.activation_func)

        if self.readout == "feat-attention":
            self.gate_nn = nn.Linear(last_hidden_dim, self.output_dim)  # feature-level gating
            self.pooling = AttentionalAggregation(self.gate_nn, self.out_nn)
        elif self.readout == "node-attention":
            self.gate_nn = nn.Linear(last_hidden_dim, 1)  # node-level gating
            self.pooling = AttentionalAggregation(self.gate_nn, self.out_nn)
        elif self.readout == "mean":
            self.pooling = Sequential("x, batch", [(self.out_nn, "x -> x"), (global_mean_pool, "x, batch -> x")])
        elif self.readout == "max":
            self.pooling = Sequential("x, batch", [(self.out_nn, "x -> x"), (global_max_pool, "x, batch -> x")])

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply the node embedding
        if self.node_embedding_dim:
            x_0 = self.node_embedding(x[:, 0].long()).squeeze()
            x_1 = x[:, 1].float().unsqueeze(1)
            x = torch.hstack((x_0, x_1))

        for layer in self.layers:
            x = layer(x, edge_index=edge_index, batch=batch)

        return self.pooling(x, batch)

    def set_params(self, **params: object) -> None:
        for parameter, value in params.items():
            setattr(self, parameter, value)
