from __future__ import annotations

from pathlib import Path

from qiskit import QuantumCircuit

from mqt.bench import get_benchmark
from mqt.predictor import Result, rl
from mqt.predictor.evaluation import create_qiskit_result, create_tket_result, evaluate_sample_circuit


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
                    [compilation_setup + "_" + device["name"] + "_" + key for device in rl.helper.get_devices()]
                )

    assert all(key in res for key in expected_keys)
    if Path(filename).exists():
        Path(filename).unlink()


def test_false_input() -> None:
    res = create_tket_result(QuantumCircuit(1000), rl.helper.get_devices()[0])
    assert isinstance(res, Result)
    assert res.compilation_time == -1.0
    assert res.fidelity == -1.0
    assert res.critical_depth == -1.0

    res = create_qiskit_result(QuantumCircuit(1000), rl.helper.get_devices()[0])
    assert isinstance(res, Result)
    assert res.compilation_time == -1.0
    assert res.fidelity == -1.0
    assert res.critical_depth == -1.0
