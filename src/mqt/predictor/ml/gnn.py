from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional
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
    zx: bool = False

    def __init__(self, **kwargs: object) -> None:
        super().__init__()
        self.set_params(**kwargs)

        if self.node_embedding_dim and self.node_embedding_dim > 1:
            self.node_embedding = nn.Embedding(self.num_node_categories, self.node_embedding_dim)
        if self.edge_embedding_dim and self.edge_embedding_dim > 1 and not self.zx:
            self.edge_embedding = nn.Embedding(self.num_edge_categories, self.edge_embedding_dim)

        if self.node_embedding_dim and self.node_embedding_dim == 1:  # one-hot encoding
            self.node_embedding = lambda x: functional.one_hot(x, num_classes=self.num_node_categories).float()
            self.node_embedding_dim = self.num_node_categories
        if self.edge_embedding_dim and self.edge_embedding_dim == 1 and not self.zx:  # one-hot encoding
            self.edge_embedding = lambda x: functional.one_hot(x, num_classes=self.num_edge_categories).float()
            self.edge_embedding_dim = self.num_edge_categories

        # hidden dimension accounting for multi-head concatenation
        if self.model == "TransformerConv" and self.concat is True:
            inner_hidden_dim = self.hidden_dim * self.heads
        else:
            inner_hidden_dim = self.hidden_dim

        if self.batch_norm:
            self.batch_norm_layer = nn.BatchNorm1d(inner_hidden_dim)
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

        edge_dim = None if self.zx else self.edge_embedding_dim or 43

        if self.model == "TransformerConv":
            self.layers = []
            for i in range(self.num_layers):
                if i == 0 and not self.zx:
                    in_channels = self.node_embedding_dim or 1  # gate
                if i == 0 and self.zx:
                    in_channels = self.node_embedding_dim or 2  # gate + phase
                if i > 0:
                    in_channels = inner_hidden_dim
                layer = Sequential(
                    "x, edge_index, edge_attr, batch",
                    [
                        (
                            TransformerConv(
                                in_channels=in_channels,
                                out_channels=self.hidden_dim,
                                edge_dim=edge_dim,
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
            last_hidden_dim = inner_hidden_dim

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
                    edge_dim=edge_dim,
                    v2=self.v2,
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
                    # edge_dim=edge_dim,
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
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Apply the node and edge embeddings
        if self.zx:
            x_0 = (
                self.node_embedding(x[:, 0].long()).squeeze()
                if self.node_embedding_dim
                else x[:, 0].float().unsqueeze(1)
            )
            x_1 = x[:, 1].float().unsqueeze(1)
            x = torch.hstack((x_0, x_1))

        if not self.zx:
            x = self.node_embedding(x.long()).squeeze() if self.node_embedding_dim else x.float()
            edge_attr = (
                self.edge_embedding(edge_attr.long()).squeeze() if self.edge_embedding_dim else edge_attr.float()
            )

        for layer in self.layers:
            x = layer(x, edge_index=edge_index, edge_attr=edge_attr if not self.zx else None, batch=batch)

        return self.pooling(x, batch)

    def set_params(self, **params: object) -> None:
        for parameter, value in params.items():
            setattr(self, parameter, value)


#
# class GraphFeaturesExtractor(BaseFeaturesExtractor):
#    def __init__(self, observation_space: gymnasium.spaces.Graph, features_dim: int = 64):
#        super(GraphFeaturesExtractor, self).__init__(observation_space, features_dim)
#
#        num_node_categories = 10 # distinct gate types (incl. 'id' and 'meas')
#        num_edge_categories = 10 # distinct wires (quantum + classical)
#        node_embedding_dim = 4 # dimension of the node embedding
#        edge_embedding_dim = 4 # dimension of the edge embedding
#        num_layers = 2 # number of neighbor aggregations
#        hidden_dim = 16 # dimension of the hidden layers
#        output_dim = 3 # dimension of the output vector
#
#        self.node_embedding = nn.Embedding(num_node_categories, node_embedding_dim)
#        self.edge_embedding = nn.Embedding(num_edge_categories, edge_embedding_dim)
#
#        self.layers = []
#        for _ in range(num_layers):
#            self.layers.append(TransformerConv(-1, hidden_dim, edge_dim=edge_embedding_dim+2))
#
#        self.gate_nn = nn.Linear(hidden_dim, 1)
#        self.nn = nn.Linear(hidden_dim, output_dim)
#        self.global_attention = GlobalAttention(self.gate_nn, self.nn)
#
#
#    def forward(self, data: torch.Tensor) -> torch.Tensor:
#        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
#
#        # Apply the node and edge embeddings
#        x = self.node_embedding(x).squeeze()
#        embedding = self.edge_embedding(edge_attr[:, 0])
#        edge_attr = torch.cat([embedding, edge_attr[:, 1:]], dim=1)
#
#        for layer in self.layers:
#            x = layer(x, edge_index, edge_attr)
#            x = nn.functional.relu(x)
#
#        # Apply a readout layer to get a single vector that represents the entire graph
#        x = self.global_attention(x, batch)
#
#        return x
#
# class GraphMaskableActorCriticPolicy(MaskableActorCriticPolicy):
#    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
#        super(GraphMaskableActorCriticPolicy, self).__init__(observation_space, action_space, lr_schedule,
#                                                             features_extractor_class=GraphFeaturesExtractor, **kwargs)
#
#
