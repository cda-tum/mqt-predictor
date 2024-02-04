from __future__ import annotations

import torch  # type: ignore[import-not-found]
import torch.nn as nn  # type: ignore[import-not-found]
import torch.nn.functional as F  # type: ignore[import-not-found]
from torch_geometric.nn import (  # type: ignore[import-not-found]
    AttentionalAggregation,
    TransformerConv,
    global_max_pool,
    global_mean_pool,
)

# import gymnasium
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class Net(nn.Module):  # type: ignore[misc]
    num_node_categories: int
    num_edge_categories: int
    node_embedding_dim: int
    edge_embedding_dim: int
    num_layers: int
    hidden_dim: int
    output_dim: int
    dropout: float
    batch_norm: bool
    activation: str
    readout: str
    heads: int
    concat: bool
    beta: float
    bias: bool
    root_weight: bool

    def __init__(self, **kwargs: object) -> None:
        super().__init__()

        defaults = {
            "num_node_categories": 0,
            "num_edge_categories": 0,
            "node_embedding_dim": 0,
            "edge_embedding_dim": 0,
            "num_layers": 1,
            "hidden_dim": 16,
            "output_dim": 7,
            "dropout": 0.0,
            "batch_norm": False,
            "activation": "relu",
            "readout": "mean",
            "heads": 1,
            "concat": True,
            "beta": 1.0,
            "bias": True,
            "root_weight": True,
        }

        # Update defaults with provided kwargs
        defaults.update(kwargs)
        self.set_params(**defaults)

        if self.node_embedding_dim:
            self.node_embedding = nn.Embedding(self.num_node_categories, self.node_embedding_dim)
        if self.edge_embedding_dim:
            self.edge_embedding = nn.Embedding(self.num_edge_categories, self.edge_embedding_dim)

        if self.activation == "relu":
            self.activation_func = nn.ReLU()
        elif self.activation == "leaky_relu":
            self.activation_func = nn.LeakyReLU()
        elif self.activation == "tanh":
            self.activation_func = nn.Tanh()
        elif self.activation == "sigmoid":
            self.activation_func = nn.Sigmoid()

        self.layers = []
        for _ in range(self.num_layers):
            self.layers.append(
                TransformerConv(
                    -1,
                    self.hidden_dim,
                    edge_dim=self.edge_embedding_dim if self.edge_embedding_dim else 1,
                    heads=self.heads,
                    concat=self.concat,
                    beta=self.beta,
                    dropout=self.dropout,
                    bias=self.bias,
                    root_weight=self.root_weight,
                )
            )

        last_hidden_dim = self.hidden_dim * self.heads if self.concat is True and self.heads > 1 else self.hidden_dim

        if self.batch_norm:
            self.batch_norm_layer = nn.BatchNorm1d(last_hidden_dim)

        self.out_nn = nn.Sequential(nn.Linear(last_hidden_dim, self.output_dim), self.activation_func)

        if self.readout == "feat-attention":
            self.gate_nn = nn.Linear(last_hidden_dim, self.output_dim)  # feature-level gating
            self.pooling = AttentionalAggregation(self.gate_nn, self.out_nn)
        elif self.readout == "node-attention":
            self.gate_nn = nn.Linear(last_hidden_dim, 1)  # node-level gating
            self.pooling = AttentionalAggregation(self.gate_nn, self.out_nn)
        elif self.readout == "mean":
            self.pooling = lambda x, batch: global_mean_pool(x, batch)
        elif self.readout == "max":
            self.pooling = lambda x, batch: global_max_pool(x, batch)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Apply the node and edge embeddings
        x = self.node_embedding(x.int()).squeeze() if self.node_embedding_dim else x.float()

        edge_attr = self.edge_embedding(edge_attr.int()).squeeze() if self.edge_embedding_dim else edge_attr.float()

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
            x = self.activation_func(x)

            if self.batch_norm:
                x = self.batch_norm_layer(x)

        # Apply a readout layer to get a single vector that represents the entire graph
        if self.readout == "mean" or self.readout == "max":
            x = self.out_nn(x)  # in "attention" case, this is done in the pooling layer

        x = self.pooling(x, batch)
        return F.sigmoid(x)

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
