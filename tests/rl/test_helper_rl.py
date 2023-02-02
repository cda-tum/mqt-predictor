from pathlib import Path

import pytest
from mqt.bench import get_benchmark
from mqt.predictor import rl
from qiskit import QuantumCircuit

NUM_ACTIONS_OPT = 12
def test_get_actions_opt() -> None:
    assert len(rl.helper.get_actions_opt()) == NUM_ACTIONS_OPT

NUM_ACTIONS_LAYOUT = 3
def test_get_actions_layout() -> None:
    assert len(rl.helper.get_actions_layout()) == NUM_ACTIONS_LAYOUT

NUM_ACTIONS_ROUTING = 4
def test_et_actions_routing() -> None:
    assert len(rl.helper.get_actions_routing()) == NUM_ACTIONS_ROUTING

NUM_ACTIONS_PLATFORM = 4
def test_get_actions_platform_selection() -> None:
    assert len(rl.helper.get_actions_platform_selection()) == NUM_ACTIONS_PLATFORM

NUM_ACTIONS_SYNTHESIS = 1
def test_get_actions_synthesis() -> None:
    assert len(rl.helper.get_actions_synthesis()) == NUM_ACTIONS_SYNTHESIS

NUM_ACTIONS_TERMINATE = 1
def test_get_action_terminate() -> None:
    assert len(rl.helper.get_action_terminate()) == NUM_ACTIONS_TERMINATE


NUM_ACTIONS_DEVICES = 5
def test_get_actions_devices() -> None:
    assert len(rl.helper.get_actions_devices()) == NUM_ACTIONS_DEVICES


def test_get_random_state_sample() -> None:
    sample = rl.helper.get_state_sample()
    assert sample
    assert isinstance(sample, QuantumCircuit)


def test_get_ibm_native_gates() -> None:
    assert rl.helper.get_ibm_native_gates() == ["rz", "sx", "x", "cx", "measure"]


def test_get_rigetti_native_gates() -> None:
    assert rl.helper.get_rigetti_native_gates() == ["rx", "rz", "cz", "measure"]


def test_get_ionq_native_gates() -> None:
    assert rl.helper.get_ionq_native_gates() == ["rxx", "rz", "ry", "rx", "measure"]


def test_get_oqc_native_gates() -> None:
    assert rl.helper.get_oqc_native_gates() == ["rz", "sx", "x", "ecr", "measure"]

NUM_CONNECTIONS_RIGETTI_M2 = 212
def test_get_rigetti_aspen_m2_map() -> None:
    assert len(rl.helper.get_rigetti_aspen_m2_map()) == NUM_CONNECTIONS_RIGETTI_M2

NUM_CONNECTIONS_IONQ_11 = 110
def test_get_ionq11_c_map() -> None:
    assert len(rl.helper.get_ionq11_c_map()) == NUM_CONNECTIONS_IONQ_11

NUM_CONNECTIONS_OQC_LUCY = 8
def test_get_cmap_oqc_lucy() -> None:
    assert len(rl.helper.get_cmap_oqc_lucy()) == NUM_CONNECTIONS_OQC_LUCY


@pytest.mark.parametrize(
    "device",
    ["ibm_washington", "ibm_montreal", "rigetti_aspen_m2", "oqc_lucy", "ionq11"],
)
def test_get_cmap_from_devicename(device: str) -> None:
    assert rl.helper.get_cmap_from_devicename(device)

NUM_FEATURES = 7
def test_create_feature_dict() -> None:
    qc = get_benchmark("dj", 1, 5)
    features = rl.helper.create_feature_dict(qc)
    assert features
    assert len(features) == NUM_FEATURES


def test_get_path_trained_model() -> None:
    path = rl.helper.get_path_trained_model()
    assert path
    assert isinstance(path, Path)


def test_get_path_training_circuits() -> None:
    path = rl.helper.get_path_training_circuits()
    assert path
    assert isinstance(path, Path)
