"""Tests for the machine learning device selection predictor module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from qiskit import QuantumCircuit

from mqt.bench.devices import get_device_by_name
from mqt.predictor import ml
from mqt.predictor.hellinger import calc_device_specific_features, hellinger_distance


# create fixture to get the predictro
@pytest.fixture
def predictor() -> ml.Predictor:
    """Return the predictor."""
    return ml.Predictor(figure_of_merit="expected_fidelity", devices=["ionq_harmony"])


@pytest.fixture
def source_path() -> Path:
    """Return the source path."""
    return Path("./test_uncompiled_circuits")


@pytest.fixture
def target_path() -> Path:
    """Return the target path."""
    return Path("./test_compiled_circuits")


def test_create_device_specific_feature_dict() -> None:
    """Test the creation of a device-specific feature vector."""
    device = get_device_by_name("iqm_adonis")
    qc = QuantumCircuit(device.num_qubits)
    for i in range(1, device.num_qubits):
        qc.cz(0, i)

    feature_vector = calc_device_specific_features(qc, device)
    expected_feat_vec = np.array([0.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.0, 5.0, 1.0, 1.0, 0.0, 2 / 5, 1 / 5, 0.0, 1 / 2])

    assert np.allclose(feature_vector, expected_feat_vec)


def test_hellinger_distance() -> None:
    """Test the calculation of the Hellinger distance."""
    p = [0.0, 1.0]
    q = [1.0, 0.0]
    distance = hellinger_distance(p, q)
    assert distance == 1


def test_hellinger_distance_error() -> None:
    """Test error during Hellinger distance calculation."""
    valid = [0.5, 0.5]
    invalid = [0.5, 0.4]

    with pytest.raises(AssertionError, match="q is not a probability distribution"):
        hellinger_distance(p=valid, q=invalid)
    with pytest.raises(AssertionError, match="p is not a probability distribution"):
        hellinger_distance(p=invalid, q=valid)


def test_train_random_forest_regressor_and_predict() -> None:
    """Test the training of the random forest regressor."""
    # Setup the training environment
    device = get_device_by_name("iqm_adonis")
    n_circuits = 16

    qc = QuantumCircuit(device.num_qubits)
    for i in range(1, device.num_qubits):
        qc.cz(0, i)

    # 1. Feature Extraction
    feature_vector = calc_device_specific_features(qc, device)
    feature_vector_list = [feature_vector] * n_circuits

    # 2. Label Generation
    rng = np.random.default_rng()
    noisy = rng.random(device.num_qubits)
    noisy /= np.sum(noisy)
    noiseless = np.zeros_like(noisy)
    noiseless[0] = 1.0
    distance_label = hellinger_distance(noisy, noiseless)
    labels_list = [distance_label] * n_circuits

    # 3. Model Training
    training_data = [(feat, label) for feat, label in zip(feature_vector_list, labels_list, strict=False)]
    trained_model = ml.train_random_forest_regressor(training_data, device, save_model=True)

    assert np.isclose(trained_model.predict([feature_vector]), distance_label)
