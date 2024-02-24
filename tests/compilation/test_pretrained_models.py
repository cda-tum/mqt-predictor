from __future__ import annotations

import os
from pathlib import Path

import pytest
from qiskit import QuantumCircuit

from mqt.bench import get_benchmark
from mqt.bench.devices import get_available_device_names
from mqt.predictor import qcompile, reward, rl
from mqt.predictor.evaluation import evaluate_sample_circuit

# only run test when executed on GitHub runner
IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(not IN_GITHUB_ACTIONS, reason="Only run this test on GitHub runner")
@pytest.mark.parametrize(
    "figure_of_merit",
    ["expected_fidelity", "critical_depth"],
)
def test_qcompile_with_pretrained_models(figure_of_merit: reward.figure_of_merit) -> None:
    qc = get_benchmark("ghz", 1, 3)
    qc_compiled, compilation_information = rl.qcompile(qc, figure_of_merit=figure_of_merit, device_name="quantinuum_h2")
    assert isinstance(qc_compiled, QuantumCircuit)
    assert compilation_information is not None


def test_qcompile() -> None:
    qc = get_benchmark("ghz", 1, 5)
    qc_compiled, compilation_information, quantum_device = qcompile(qc)
    assert quantum_device in get_available_device_names()
    assert qc_compiled.layout is not None
    assert len(qc_compiled) > 0


def test_evaluate_sample_circuit() -> None:
    qc = get_benchmark("ghz", 1, 3)
    filename = "test_3.qasm"
    qc.qasm(filename=filename)
    res = evaluate_sample_circuit(filename)
    expected_keys = []
    for compilation_setup in ["qiskit", "tket", "mqt-predictor_expected_fidelity", "mqt-predictor_critical_depth"]:
        for key in ["time", "expected_fidelity", "critical_depth"]:
            if "mqt-predictor" in compilation_setup:
                expected_keys.append(compilation_setup + "_" + key)
            else:
                expected_keys.extend(
                    [compilation_setup + "_" + name + "_" + key for name in get_available_device_names()]
                )

    assert all(key in res for key in expected_keys)
    if Path(filename).exists():
        Path(filename).unlink()
