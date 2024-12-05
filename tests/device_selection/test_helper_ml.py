"""Tests for the helper functions in the ml module."""

from __future__ import annotations

import pytest

from mqt.bench import benchmark_generator
from mqt.predictor import ml


def test_load_and_save_training_data() -> None:
    """Test the loading and saving of the training data."""
    with pytest.raises(FileNotFoundError, match="Training data not found. Please run the training script first."):
        ml.helper.load_training_data("false_input")  # type: ignore[arg-type]

    training_data, names_list, scores_list = ml.helper.load_training_data()
    assert training_data is not None
    assert names_list is not None
    assert scores_list is not None
    ml.helper.save_training_data(training_data, names_list, scores_list, "expected_fidelity")


def test_create_feature_dict() -> None:
    """Test the creation of a feature dictionary."""
    qc = benchmark_generator.get_benchmark("dj", 1, 3)
    feature_vector = ml.helper.create_feature_dict(qc)
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


def test_get_path_results() -> None:
    """Test the retrieval of the path to the results."""
    for get_ghz_path_results in (True, False):
        path = ml.helper.get_path_results(ghz_results=get_ghz_path_results)
        assert path.exists()
