from pathlib import Path

from mqt.bench import get_benchmark
from mqt.predictor import rl
from qiskit import QuantumCircuit


def test_get_random_state_sample() -> None:
    sample = rl.helper.get_state_sample()
    assert sample
    assert isinstance(sample, QuantumCircuit)


NUM_FEATURES = 7


def test_create_feature_dict() -> None:
    qc = get_benchmark("dj", 1, 5)
    features = rl.helper.create_feature_dict(qc)
    assert features
    assert len(features) == NUM_FEATURES


def test_get_path_trained_model() -> None:
    path = rl.helper.get_path_trained_model()
    assert path.exists()
    assert isinstance(path, Path)


def test_get_path_training_circuits() -> None:
    path = rl.helper.get_path_training_circuits()
    assert path.exists()
    assert isinstance(path, Path)
