from __future__ import annotations

import torch  # type: ignore[import-not-found]
from torch_geometric.data import Data, DataLoader, Dataset  # type: ignore[import-not-found]

from mqt.predictor.ml.GNN import Net


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

    def __init__(
        self,
        optimizer: str = "adam",
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 10,
        num_node_categories: int = 2,  # 42,  # distinct gate types (incl. 'id' and 'meas')
        num_edge_categories: int = 2,  # 100,  # distinct wires (quantum + classical)
        node_embedding_dim: int = 4,  # dimension of the node embedding
        edge_embedding_dim: int = 4,  # dimension of the edge embedding
        num_layers: int = 2,  # number of nei aggregations
        hidden_dim: int = 16,  # dimension of the hidden layers
        output_dim: int = 1,  # dimension of the output vector
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
        )
        if optimizer == "adam":
            self.optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=5e-4)
        elif optimizer == "sgd":
            self.optim = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)

    def get_params(self, deep: bool = False) -> dict[str, object]:
        print(deep)
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
                loss = torch.nn.MSELoss()(out, batch.y)
                loss.backward()
                self.optim.step()
        return

    def predict(self, data: Data) -> torch.Tensor:
        self.model.eval()
        logits = self.model.forward(data)
        return logits.argmax(dim=-1)

    def score(self, X: Data) -> float:
        self.predict(X)
        return 1.0
