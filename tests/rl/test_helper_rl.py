from pathlib import Path

import pytest
from mqt.bench import get_benchmark
from mqt.bench.qiskit_helper import get_native_gates
from mqt.bench.utils import get_cmap_from_devicename
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


NUM_ACTIONS_SYNTHESIS = 1


def test_get_actions_synthesis() -> None:
    assert len(rl.helper.get_actions_synthesis()) == NUM_ACTIONS_SYNTHESIS


NUM_ACTIONS_TERMINATE = 1


def test_get_action_terminate() -> None:
    assert len(rl.helper.get_action_terminate()) == NUM_ACTIONS_TERMINATE


NUM_ACTIONS_DEVICES = 7


def test_get_actions_devices() -> None:
    assert len(rl.helper.get_devices()) == NUM_ACTIONS_DEVICES


def test_get_random_state_sample() -> None:
    sample = rl.helper.get_state_sample()
    assert sample
    assert isinstance(sample, QuantumCircuit)


@pytest.mark.parametrize(
    ("device", "gate_set"),
    [
        ("ibm", ["rz", "sx", "x", "cx", "measure"]),
        ("ionq", ["rxx", "rz", "ry", "rx", "measure"]),
        ("rigetti", ["rx", "rz", "cz", "measure"]),
        ("oqc", ["rz", "sx", "x", "ecr", "measure"]),
        ("quantinuum", ["rzz", "rz", "ry", "rx", "measure"]),
    ],
)
def test_get_native_gatesets(device: str, gate_set: list[str]) -> None:
    assert get_native_gates(device) == gate_set


NUM_CONNECTIONS_RIGETTI_M2 = 212


def test_get_rigetti_aspen_m2_map() -> None:
    assert len(get_cmap_from_devicename("rigetti_aspen_m2")) == NUM_CONNECTIONS_RIGETTI_M2


NUM_CONNECTIONS_IONQ_HARMONY = 110


def test_get_ionq_harmony_c_map() -> None:
    assert len(get_cmap_from_devicename("ionq_harmony")) == NUM_CONNECTIONS_IONQ_HARMONY


NUM_CONNECTIONS_OQC_LUCY = 8


def test_get_cmap_oqc_lucy() -> None:
    assert len(get_cmap_from_devicename("oqc_lucy")) == NUM_CONNECTIONS_OQC_LUCY


@pytest.mark.parametrize(
    "device",
    ["ibm_washington", "ibm_montreal", "rigetti_aspen_m2", "oqc_lucy", "ionq_harmony", "ionq_aria1", "quantinuum_h2"],
)
def test_get_cmap_from_devicename(device: str) -> None:
    assert get_cmap_from_devicename(device)


@pytest.mark.parametrize(
    "platform",
    ["ibm", "rigetti", "oqc", "ionq", "quantinuum"],
)
def test_get_native_gates_from_platform_name(platform: str) -> None:
    assert get_native_gates(platform)


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
