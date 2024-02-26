from __future__ import annotations

from pathlib import Path

import numpy as np

from mqt.bench import get_benchmark
from mqt.predictor import rl


def test_create_feature_dict() -> None:
    qc = get_benchmark("dj", 1, 5)
    features = rl.helper.create_feature_dict(qc)
    for feature in features.values():
        assert isinstance(feature, np.ndarray | int)


def test_get_path_trained_model() -> None:
    path = rl.helper.get_path_trained_model()
    assert path.exists()
    assert isinstance(path, Path)


def test_get_path_training_circuits() -> None:
    path = rl.helper.get_path_training_circuits()
    assert path.exists()
    assert isinstance(path, Path)
