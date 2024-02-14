from __future__ import annotations

from typing import TYPE_CHECKING

import torch  # type: ignore[import-not-found]
from torch_geometric.loader import DataLoader  # type: ignore[import-not-found]
from torch_geometric.nn import VGAE  # type: ignore[import-not-found]

from mqt.predictor.ml.GNN import Net

if TYPE_CHECKING:
    from torch_geometric.data import Dataset  # type: ignore[import-not-found]


class GNNAutoencoder:
    optimizer: str
    learning_rate: float
    batch_size: int
    epochs: int
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
    beta: bool
    bias: bool
    root_weight: bool
    model: str
    jk: str
    v2: bool
    output_mask: torch.Tensor

    def __init__(self, **kwargs: object) -> None:
        self.defaults = {
            "optimizer": "adam",
            "learning_rate": 1e-3,
            "batch_size": 64,
            "epochs": 10,
            "num_node_categories": 42,
            "num_edge_categories": 2,
            "node_embedding_dim": 4,
            "edge_embedding_dim": 4,
            "num_layers": 2,
            "hidden_dim": 16,
            "output_dim": 7,
            "dropout": 0.0,
            "batch_norm": False,
            "activation": "relu",
            "readout": "mean",
            "heads": 1,
            "concat": True,
            "beta": False,
            "bias": True,
            "root_weight": True,
            "model": "TransformerConv",
            "jk": "last",
            "v2": True,
            "output_mask": torch.tensor([True, True, True, True, True, True, True]),
        }

        # Update defaults with provided kwargs
        self.set_params(**kwargs)

    def get_params(self, deep: bool = False) -> dict[str, object]:
        if deep:
            print("deep copy not implemented")
        return dict(self.defaults.items())

    def set_params(self, **params: object) -> GNNAutoencoder:
        # Update defaults with provided kwargs
        self.defaults.update(params)
        for parameter, value in self.defaults.items():
            setattr(self, parameter, value)

        # define the model and autoencoder
        self.gnn = Net(**params)
        self.vgae = VGAE(self.gnn)

        # define the optimizer
        if self.optimizer == "adam":
            self.optim = torch.optim.Adam(self.vgae.parameters(), lr=self.learning_rate, weight_decay=5e-4)
        elif self.optimizer == "sgd":
            self.optim = torch.optim.SGD(self.vgae.parameters(), lr=self.learning_rate, momentum=0.9)

        return self

    def fit(self, dataset: Dataset) -> None:
        self.vgae.train()
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for _ in range(self.epochs):
            for batch in loader:
                self.optim.zero_grad()
                z = self.vgae.encode(batch)
                loss = self.vgae.recon_loss(z, batch.edge_index)
                if isinstance(self.vgae, VGAE):
                    loss = loss + self.vgae.kl_loss()
                loss.backward()
                self.optim.step()
        return

    def predict(self, dataset: Dataset) -> torch.Tensor:
        self.vgae.eval()
        out = [self.vgae.encode(data.x, data.edge_index) for data in dataset]
        return torch.stack(out)

    def score(self, dataset: Dataset) -> float:
        z_pred = self.predict(dataset)
        z_true = torch.stack([data.y for data in dataset])
        mse = torch.nn.functional.mse_loss(z_pred, z_true)
        return -mse.item()
