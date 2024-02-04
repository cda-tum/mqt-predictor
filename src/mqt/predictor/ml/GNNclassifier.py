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
    beta: float
    bias: bool
    root_weight: bool

    def __init__(self, **kwargs: object) -> None:
        defaults = {
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
            "beta": 1.0,
            "bias": True,
            "root_weight": True,
        }

        # Update defaults with provided kwargs
        defaults.update(kwargs)
        self.set_params(**defaults)

        if self.optimizer == "adam":
            self.optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=5e-4)
        elif self.optimizer == "sgd":
            self.optim = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)

    def get_params(self, deep: bool = False) -> dict[str, object]:
        if deep:
            print("deep copy not implemented")
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_") and not callable(v)}

    def set_params(self, **params: object) -> GNNClassifier:
        for parameter, value in params.items():
            setattr(self, parameter, value)
        self.model = Net(**params)
        return self

    def fit(self, dataset: Dataset) -> None:
        self.model.train()
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for _ in range(self.epochs):
            for batch in loader:
                self.optim.zero_grad()
                out = self.model.forward(batch)
                loss = torch.nn.MSELoss()(out, batch.y.view(-1, self.output_dim))
                loss.backward()
                self.optim.step()
        return

    def predict(self, dataset: Dataset) -> torch.Tensor:
        self.model.eval()
        out = [self.model.forward(data) for data in dataset]
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
        self.models = [Net(**kwargs) for _ in range(self.global_output_dim)]  # type: ignore[call-overload]

    def set_params(self, **params: object) -> MultiGNNClassifier:
        super().set_params(**params)

        self.global_output_dim = params["output_dim"]
        # one model for each output
        params["output_dim"] = 1
        self.models = [Net(**params) for _ in range(self.global_output_dim)]  # type: ignore[call-overload]
        return self

    def fit(self, dataset: Dataset) -> None:
        for m in self.models:
            m.train()

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for _ in range(self.epochs):
            for batch in loader:
                self.optim.zero_grad()
                out = torch.hstack([model.forward(batch) for model in self.models])  # dim: (batch_size, len(models)=7)
                loss = torch.nn.MSELoss()(out, batch.y.view(-1, len(self.models)))
                loss.backward()
                self.optim.step()
        return

    def predict(self, dataset: Dataset) -> torch.Tensor:
        pred = []
        for m in self.models:
            m.eval()

        for data in dataset:
            out = [model.forward(data) for model in self.models]
            pred.append(torch.tensor(out).argmax())

        return torch.stack(pred)
