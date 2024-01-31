from __future__ import annotations

import torch  # type: ignore[import-not-found]
import torch.nn as nn  # type: ignore[import-not-found]
from torch_geometric.nn import (  # type: ignore[import-not-found]
    AttentionalAggregation,
    TransformerConv,
)

# import gymnasium
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class Net(torch.nn.Module):  # type: ignore[misc]
    def __init__(
        self,
        num_node_categories: int,
        num_edge_categories: int,
        node_embedding_dim: int,
        edge_embedding_dim: int,
        num_layers: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
        batch_norm: bool,
    ) -> None:
        super().__init__()

        self.node_embedding_dim = node_embedding_dim
        self.edge_embedding_dim = edge_embedding_dim
        self.dropout = dropout
        self.batch_norm = batch_norm

        if node_embedding_dim:
            self.node_embedding = nn.Embedding(num_node_categories, node_embedding_dim)
        if edge_embedding_dim:
            self.edge_embedding = nn.Embedding(num_edge_categories, edge_embedding_dim)

        if self.dropout > 0:
            self.dropout_layer = torch.nn.Dropout(self.dropout)

        if self.batch_norm:
            self.batch_norm_layer = torch.nn.BatchNorm1d(hidden_dim)

        self.layers = []
        for _ in range(num_layers - 1):
            self.layers.append(
                TransformerConv(
                    -1, hidden_dim, edge_dim=edge_embedding_dim if edge_embedding_dim else num_edge_categories
                )
            )

        self.gate_nn = torch.nn.Linear(hidden_dim, 1)
        self.nn = torch.nn.Linear(hidden_dim, output_dim)
        self.attention = AttentionalAggregation(self.gate_nn, self.nn)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Apply the node and edge embeddings
        if self.node_embedding_dim:
            x = self.node_embedding(x.int()).squeeze()
        if self.edge_embedding_dim:
            edge_attr = self.edge_embedding(edge_attr.int()).squeeze()
        # embedding = self.edge_embedding(edge_attr[:, 0].int())
        # edge_attr = torch.cat([embedding, edge_attr[:, 1:]], dim=1)

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
            x = torch.nn.functional.relu(x)
            if self.dropout > 0:
                x = self.dropout_layer(x)
            if self.batch_norm:
                x = self.batch_norm_layer(x)

        # Apply a readout layer to get a single vector that represents the entire graph
        return self.attention(x, batch)


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
#        self.gate_nn = torch.nn.Linear(hidden_dim, 1)
#        self.nn = torch.nn.Linear(hidden_dim, output_dim)
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
#            x = torch.nn.functional.relu(x)
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
