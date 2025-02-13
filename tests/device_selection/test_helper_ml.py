"""Tests for the helper functions in the ml module."""

from __future__ import annotations

from mqt.bench import benchmark_generator
from mqt.bench.devices import get_device_by_name
from mqt.predictor import ml


def test_create_feature_vector() -> None:
    """Test the creation of a feature dictionary."""
    qc = benchmark_generator.get_benchmark("dj", 1, 3)
    feature_vector = ml.helper.create_feature_vector(qc)
    assert feature_vector is not None


def test_create_device_specific_feature_dict() -> None:
    """Test the creation of a device specific feature dictionary."""
    device = get_device_by_name("iqm_adonis")
    qc = benchmark_generator.get_benchmark("dj", 1, 3)
    feature_vector = ml.helper.calc_device_specific_features(qc, device)
    assert feature_vector is not None


def test_get_openqasm_gates() -> None:
    """Test the retrieval of the OpenQASM gates."""
    assert ml.helper.get_openqasm_gates() is not None


def test_get_path_training_circuits() -> None:
    """Test the retrieval of the path to the training circuits."""
    path = ml.helper.get_path_training_circuits()
    assert path.exists()


def test_get_path_training_circuits_compiled() -> None:
    """Test the retrieval of the path to the compiled training circuits."""
    path = ml.helper.get_path_training_circuits_compiled()
    assert path.exists()


def test_get_path_training_data() -> None:
    """Test the retrieval of the path to the training data."""
    path = ml.helper.get_path_training_data()
    assert path.exists()
