from __future__ import annotations

from pathlib import Path

import numpy as np
from qiskit import transpile
from qiskit.transpiler import PassManager

from mqt.bench import get_benchmark
from mqt.bench.devices import get_device_by_name
from mqt.predictor import rl


def test_create_feature_dict() -> None:
    qc = get_benchmark("dj", 1, 5)
    features = rl.helper.create_feature_dict(qc)
    for feature in features.values():
        assert isinstance(feature, np.ndarray | int)


def test_get_path_trained_model() -> None:
    path = rl.helper.get_path_trained_model()
    assert path.exists()
    assert isinstance(path, Path)


def test_get_path_training_circuits() -> None:
    path = rl.helper.get_path_training_circuits()
    assert path.exists()
    assert isinstance(path, Path)


def test_VF2_layout_and_postlayout() -> None:
    qc = get_benchmark("ghz", 1, 3)
    # qc_transpiled = transpile(qc, basis_gates=dev.basis_gates, coupling_map=dev.coupling_map)

    for dev in [get_device_by_name("ibm_montreal"), get_device_by_name("ionq_harmony")]:
        layout_pass = None
        for layout_action in rl.helper.get_actions_layout():
            if layout_action["name"] == "VF2Layout":
                layout_pass = layout_action["transpile_pass"](dev)
                break
        pm = PassManager(layout_pass)
        pm.run(qc)
        assert pm.property_set["VF2Layout_stop_reason"] is not None

    qc_transpiled = transpile(qc, basis_gates=dev.basis_gates, coupling_map=dev.coupling_map)
    assert qc_transpiled.layout is not None

    dev_success = get_device_by_name("ibm_montreal")
    post_layout_pass = None
    for layout_action in rl.helper.get_actions_final_optimization():
        if layout_action["name"] == "VF2PostLayout":
            post_layout_pass = layout_action["transpile_pass"](dev_success)
            break

    pm = PassManager(post_layout_pass)
    pm.run(qc_transpiled)
    assert pm.property_set["VF2PostLayout_stop_reason"] is not None
