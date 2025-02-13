"""This module contains the functions to calculate the reward of a quantum circuit on a given device."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqt.bench.devices import Device


def hellinger_distance(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    """Calculates the Hellinger distance between two probability distributions."""
    assert np.isclose(np.sum(p), 1, 0.05), "p is not a probability distribution"
    assert np.isclose(np.sum(q), 1, 0.05), "q is not a probability distribution"

    return float((1 / np.sqrt(2)) * np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))


def get_hellinger_model_path(device: Device = None) -> Path:
    """Returns the path to the trained model folder resulting from the machine learning training."""
    training_data_path = Path(str(resources.files("mqt.predictor"))) / "ml" / "training_data"
    return training_data_path / "trained_model" / ("trained_hellinger_distance_regressor_" + device.name + ".joblib")


def hellinger_model_available(device: Device) -> bool:
    """Check if a trained model to estimate the Hellinger distance is available for the device."""
    path = get_hellinger_model_path(device)
    return bool(path.is_file())
