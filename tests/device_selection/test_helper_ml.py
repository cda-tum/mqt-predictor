"""Tests for the helper functions in the ml module."""

from __future__ import annotations

import re

import numpy as np
import pytest
from qiskit import QuantumRegister
from qiskit.circuit.quantumcircuit import Qubit

from mqt.bench import benchmark_generator
from mqt.bench.devices import get_device_by_name
from mqt.predictor import ml


def test_load_and_save_training_data() -> None:
    """Test the loading and saving of the training data."""
    with pytest.raises(
        FileNotFoundError, match=re.escape("Training data not found. Please run the training script first.")
    ):
        ml.helper.load_training_data("false_input")  # type: ignore[arg-type]

    training_data, names_list, scores_list = ml.helper.load_training_data()
    assert training_data is not None
    assert names_list is not None
    assert scores_list is not None
    ml.helper.save_training_data(training_data, names_list, scores_list, "test")
    for file in ["training_data_test.npy", "names_list_test.npy", "scores_list_test.npy"]:
        path = ml.helper.get_path_training_data() / "training_data_aggregated" / file
        assert path.exists()
        path.unlink()


def test_create_feature_dict() -> None:
    """Test the creation of a feature dictionary."""
    qc = benchmark_generator.get_benchmark("dj", 1, 3)
    feature_vector = ml.helper.create_feature_dict(qc)
    assert feature_vector is not None


def test_create_device_specific_feature_dict() -> None:
    """Test the creation of a device specific feature dictionary."""
    qc = benchmark_generator.get_benchmark("dj", 1, 3)
    device = get_device_by_name("ibm_montreal")
    feature_vector = ml.helper.calc_device_specific_features(qc, device)

    thesis_dict = {
        "id": 0,
        "rz": 0,
        "sx": 0,
        "x": 0,
        "cx": 2,
        "measure": 2,
        "barrier": 1,
        Qubit(QuantumRegister(3, "q"), 0): 1,
        Qubit(QuantumRegister(3, "q"), 1): 1,
        Qubit(QuantumRegister(3, "q"), 2): 1,
        "depth": 5.0,
        "num_qubits": 3,
        "program_communication": 0.6666666666666666,
        "critical_depth": 1.0,
        "entanglement_ratio": 0.2857142857142857,
        "parallelism": 0.19999999999999996,
        "liveness": np.float64(0.7333333333333333),
        "directed_program_communication": 0.3333333333333333,
        "single_qubit_gates_per_layer": 0.4166666666666667,
        "multi_qubit_gates_per_layer": 0.5,
    }

    # Verify equality of the feature vectors
    for key, val in thesis_dict.items():
        assert feature_vector[key] == val


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
