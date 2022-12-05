import os
from pathlib import Path

import pytest
from mqt.bench import get_benchmark
from qiskit import QuantumCircuit

from mqt.predictor import RL_utils


def test_get_actions_opt():
    assert len(RL_utils.get_actions_opt()) == 12


def test_get_actions_layout():
    assert len(RL_utils.get_actions_layout()) == 3


def test_et_actions_routing():
    assert len(RL_utils.get_actions_routing()) == 4


def test_get_actions_platform_selection():
    assert len(RL_utils.get_actions_platform_selection()) == 4


def test_get_actions_synthesis():
    assert len(RL_utils.get_actions_synthesis()) == 1


def test_get_action_terminate():
    assert len(RL_utils.get_action_terminate()) == 1


def test_get_actions_devices():
    assert len(RL_utils.get_actions_devices()) == 5


@pytest.mark.skipif(
    os.getenv("skip_optional_tests"), reason="QASM files are not provided."
)
def test_get_random_state_sample():
    sample = RL_utils.get_random_state_sample()
    assert sample and isinstance(sample, QuantumCircuit)


def test_get_ibm_native_gates():
    assert RL_utils.get_ibm_native_gates() == ["rz", "sx", "x", "cx", "measure"]


def test_get_rigetti_native_gates():
    assert RL_utils.get_rigetti_native_gates() == ["rx", "rz", "cz", "measure"]


def test_get_ionq_native_gates():
    assert RL_utils.get_ionq_native_gates() == ["rxx", "rz", "ry", "rx", "measure"]


def get_oqc_native_gates():
    assert RL_utils.get_oqc_native_gates() == ["rz", "sx", "x", "ecr", "measure"]


def test_get_rigetti_aspen_m2_map():
    assert len(RL_utils.get_rigetti_aspen_m2_map()) == 212


def test_get_ionq11_c_map():
    assert len(RL_utils.get_ionq11_c_map()) == 110


def test_get_cmap_oqc_lucy():
    assert len(RL_utils.get_cmap_oqc_lucy()) == 8


@pytest.mark.parametrize(
    "device",
    ["ibm_washington", "ibm_montreal", "rigetti_aspen_m2", "oqc_lucy", "ionq11"],
)
def get_cmap_from_devicename(device: str):
    assert RL_utils.get_cmap_from_devicename(device)


def test_create_feature_dict_RL():
    qc = get_benchmark("dj", 1, 5)
    features = RL_utils.create_feature_dict_RL(qc)
    assert features and len(features) == 7


def test_get_path_trained_model_RL():
    path = RL_utils.get_path_trained_model_RL()
    assert path and isinstance(path, Path)


def test_get_path_training_circuits_RL():
    path = RL_utils.get_path_training_circuits_RL()
    assert path and isinstance(path, Path)
