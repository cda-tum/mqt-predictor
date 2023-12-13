from __future__ import annotations

from pathlib import Path

import pytest

from mqt.bench import get_benchmark
from mqt.predictor import rl


def test_get_actions_opt() -> None:
    assert len(rl.helper.get_actions_opt()) == rl.helper.NUM_ACTIONS_OPT


def test_get_actions_layout() -> None:
    assert len(rl.helper.get_actions_layout()) == rl.helper.NUM_ACTIONS_LAYOUT


def test_et_actions_routing() -> None:
    assert len(rl.helper.get_actions_routing()) == rl.helper.NUM_ACTIONS_ROUTING


def test_get_actions_synthesis() -> None:
    assert len(rl.helper.get_actions_synthesis()) == rl.helper.NUM_ACTIONS_SYNTHESIS


def test_get_action_terminate() -> None:
    assert len(rl.helper.get_action_terminate()) == rl.helper.NUM_ACTIONS_TERMINATE


def test_get_actions_devices() -> None:
    assert len(rl.helper.get_devices()) == rl.helper.NUM_ACTIONS_DEVICES

    with pytest.raises(RuntimeError):
        rl.helper.get_device("false_input")


def test_get_device_index_of_device_false_input() -> None:
    with pytest.raises(RuntimeError):
        rl.helper.get_device_index_of_device("false_input")


def test_get_actions_mapping() -> None:
    assert len(rl.helper.get_actions_mapping()) == rl.helper.NUM_ACTIONS_MAPPING


@pytest.mark.parametrize(
    "device",
    ["ibm_washington", "ibm_montreal", "rigetti_aspen_m2", "oqc_lucy", "ionq_harmony", "ionq_aria1", "quantinuum_h2"],
)
def test_get_device(device: str) -> None:
    assert rl.helper.get_device(device)


def test_create_feature_dict() -> None:
    qc = get_benchmark("dj", 1, 5)
    features = rl.helper.create_feature_dict(qc)
    assert features
    assert len(features) == rl.helper.NUM_FEATURE_VECTOR_ELEMENTS


def test_get_path_trained_model() -> None:
    path = rl.helper.get_path_trained_model()
    assert path.exists()
    assert isinstance(path, Path)


def test_get_path_training_circuits() -> None:
    path = rl.helper.get_path_training_circuits()
    assert path.exists()
    assert isinstance(path, Path)
