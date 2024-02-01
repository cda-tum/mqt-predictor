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

    def __init__(
        self,
        optimizer: str = "adam",
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 10,
        num_node_categories: int = 42,  # distinct gate types (incl. 'id' and 'meas')
        num_edge_categories: int = 2,  # is_control. or 100,  #for  distinct wires (quantum + classical)
        node_embedding_dim: int = 4,  # dimension of the node embedding
        edge_embedding_dim: int = 4,  # dimension of the edge embedding
        num_layers: int = 2,  # number of nei aggregations
        hidden_dim: int = 16,  # dimension of the hidden layers
        output_dim: int = 7,  # dimension of the output vector
        dropout: float = 0.0,
        batch_norm: bool = False,
        activation: str = "relu",
        readout: str = "mean",
    ) -> None:
        self.set_params(
            optimizer=optimizer,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            num_node_categories=num_node_categories,
            num_edge_categories=num_edge_categories,
            node_embedding_dim=node_embedding_dim,
            edge_embedding_dim=edge_embedding_dim,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            batch_norm=batch_norm,
            activation=activation,
            readout=readout,
        )

        # Initialize the model
        self.model = Net(
            num_node_categories,
            num_edge_categories,
            node_embedding_dim,
            edge_embedding_dim,
            num_layers,
            hidden_dim,
            output_dim,
            dropout=dropout,
            batch_norm=batch_norm,
            activation=activation,
            readout=readout,
        )
        if optimizer == "adam":
            self.optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=5e-4)
        elif optimizer == "sgd":
            self.optim = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)

    def get_params(self, deep: bool = False) -> dict[str, object]:
        if deep:
            print("deep copy not implemented")
        return {
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "num_node_categories": self.num_node_categories,
            "num_edge_categories": self.num_edge_categories,
            "node_embedding_dim": self.node_embedding_dim,
            "edge_embedding_dim": self.edge_embedding_dim,
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "dropout": self.dropout,
            "batch_norm": self.batch_norm,
        }

    def set_params(self, **params: object) -> GNNClassifier:
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self

    def fit(self, dataset: Dataset) -> None:
        self.model.train()
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for _ in range(self.epochs):
            for batch in loader:
                self.optim.zero_grad()
                out = self.model.forward(batch)
                loss = torch.nn.MSELoss()(out.flatten(), batch.y)
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
    def __init__(
        self,
        optimizer: str = "adam",
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 10,
        num_node_categories: int = 42,  # distinct gate types (incl. 'id' and 'meas')
        num_edge_categories: int = 2,  # is_control. or 100,  #for  distinct wires (quantum + classical)
        node_embedding_dim: int = 4,  # dimension of the node embedding
        edge_embedding_dim: int = 4,  # dimension of the edge embedding
        num_layers: int = 2,  # number of nei aggregations
        hidden_dim: int = 16,  # dimension of the hidden layers
        output_dim: int = 7,  # dimension of the output vector
        dropout: float = 0.0,
        batch_norm: bool = False,
        activation: str = "relu",
        readout: str = "mean",
    ) -> None:
        super().__init__(
            optimizer,
            learning_rate,
            batch_size,
            epochs,
            num_node_categories,
            num_edge_categories,
            node_embedding_dim,
            edge_embedding_dim,
            num_layers,
            hidden_dim,
            output_dim,
            dropout,
            batch_norm,
            activation,
            readout,
        )

        # one model for each output
        self.models = [
            Net(
                num_node_categories,
                num_edge_categories,
                node_embedding_dim,
                edge_embedding_dim,
                num_layers,
                hidden_dim,
                output_dim=1,
                dropout=dropout,
                batch_norm=batch_norm,
                activation=activation,
                readout=readout,
            )
            for _ in range(output_dim)
        ]

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
