from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mqt.bench import get_benchmark
from mqt.predictor import rl


def test_get_device_false_input() -> None:
    with pytest.raises(RuntimeError):
        rl.helper.get_device("false_input")


def test_get_device_index_of_device_false_input() -> None:
    with pytest.raises(RuntimeError):
        rl.helper.get_device_index_of_device("false_input")


@pytest.mark.parametrize(
    "device",
    ["ibm_washington", "ibm_montreal", "rigetti_aspen_m2", "oqc_lucy", "ionq_harmony", "ionq_aria1", "quantinuum_h2"],
)
def test_get_device(device: str) -> None:
    assert rl.helper.get_device(device)


def test_create_feature_dict() -> None:
    qc = get_benchmark("dj", 1, 5)
    features = rl.helper.create_feature_dict(qc)
    for feature in features.values():
        assert isinstance(feature, (np.ndarray, int))


def test_get_path_trained_model() -> None:
    path = rl.helper.get_path_trained_model()
    assert path.exists()
    assert isinstance(path, Path)


def test_get_path_training_circuits() -> None:
    path = rl.helper.get_path_training_circuits()
    assert path.exists()
    assert isinstance(path, Path)
