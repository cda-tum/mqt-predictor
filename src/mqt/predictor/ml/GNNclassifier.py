from __future__ import annotations

from typing import TYPE_CHECKING

import torch  # type: ignore[import-not-found]
from torch_geometric.loader import DataLoader  # type: ignore[import-not-found]

from mqt.predictor.ml.GNN import Net

if TYPE_CHECKING:
    from torch_geometric.data import Dataset  # type: ignore[import-not-found]


class GNNClassifier:
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
        }

        # Update defaults with provided kwargs
        self.defaults.update(kwargs)
        self.set_params(**self.defaults)

    def get_params(self, deep: bool = False) -> dict[str, object]:
        if deep:
            print("deep copy not implemented")
        return dict(self.defaults.items())

    def set_params(self, **params: object) -> GNNClassifier:
        self.defaults.update(params)
        for parameter, value in self.defaults.items():
            setattr(self, parameter, value)

        self.gnn = Net(**params)
        if self.optimizer == "adam":
            self.optim = torch.optim.Adam(self.gnn.parameters(), lr=self.learning_rate, weight_decay=5e-4)
        elif self.optimizer == "sgd":
            self.optim = torch.optim.SGD(self.gnn.parameters(), lr=self.learning_rate, momentum=0.9)
        return self

    def fit(self, dataset: Dataset) -> None:
        self.gnn.train()
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for _ in range(self.epochs):
            for batch in loader:
                self.optim.zero_grad()
                out = self.gnn.forward(batch)
                out = torch.nn.functional.softmax(out, dim=1)  # normalize the outputs to probabilities
                target = torch.nn.functional.softmax(
                    batch.y.view(-1, self.output_dim), dim=1
                )  # normalize the targets to probabilities
                loss = torch.nn.KLDivLoss()(out.log(), target)  # compute the KL Divergence loss
                loss.backward()
                self.optim.step()
        return

    def predict(self, dataset: Dataset) -> torch.Tensor:
        self.gnn.eval()
        out = [self.gnn.forward(data) for data in dataset]
        return torch.stack([o.argmax() for o in out])

    def score(self, dataset: Dataset) -> float:
        pred = self.predict(dataset)
        labels = torch.stack([data.y for data in dataset]).argmax(dim=1)
        correct = pred.eq(labels).sum().item()
        total = len(dataset)
        return int(correct) / total


class MultiGNNClassifier(GNNClassifier):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)

        self.global_output_dim = kwargs["output_dim"]
        # one model for each output
        kwargs["output_dim"] = 1
        self.gnns = [Net(**kwargs) for _ in range(self.global_output_dim)]  # type: ignore[call-overload]

    def set_params(self, **params: object) -> MultiGNNClassifier:
        super().set_params(**params)

        self.global_output_dim = params["output_dim"]
        # one model for each output
        params["output_dim"] = 1
        self.gnns = [Net(**params) for _ in range(self.global_output_dim)]  # type: ignore[call-overload]
        return self

    def fit(self, dataset: Dataset) -> None:
        for m in self.gnns:
            m.train()

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for _ in range(self.epochs):
            for batch in loader:
                self.optim.zero_grad()
                out = torch.hstack([gnn.forward(batch) for gnn in self.gnns])  # dim: (batch_size, len(models)=7)
                loss = torch.nn.MSELoss()(out, batch.y.view(-1, len(self.gnns)))
                loss.backward()
                self.optim.step()
        return

    def predict(self, dataset: Dataset) -> torch.Tensor:
        pred = []
        for m in self.gnns:
            m.eval()

        for data in dataset:
            out = [gnn.forward(data) for gnn in self.gnns]
            pred.append(torch.tensor(out).argmax())

        return torch.stack(pred)
