from __future__ import annotations

from qiskit import QuantumCircuit

from mqt.bench.devices import get_available_devices
from mqt.predictor import Result
from mqt.predictor.evaluation import create_qiskit_result, create_tket_result


def test_create_result() -> None:
    devices = get_available_devices()
    assert devices[0].num_qubits > 10
    qc = QuantumCircuit(10)
    for i in range(10):
        qc.id(i)
    res = create_tket_result(qc, devices[0])
    assert isinstance(res, Result)
    assert res.compilation_time >= 0.0
    assert res.fidelity >= 0.0
    assert res.critical_depth >= 0.0

    res = create_qiskit_result(qc, devices[0])
    assert isinstance(res, Result)
    assert res.compilation_time >= 0.0
    assert res.fidelity >= 0.0
    assert res.critical_depth >= 0.0


def test_false_input() -> None:
    devices = get_available_devices()
    res = create_tket_result(QuantumCircuit(1000), devices[0])
    assert isinstance(res, Result)
    assert res.compilation_time == -1.0
    assert res.fidelity == -1.0
    assert res.critical_depth == -1.0

    res = create_qiskit_result(QuantumCircuit(1000), devices[0])
    assert isinstance(res, Result)
    assert res.compilation_time == -1.0
    assert res.fidelity == -1.0
    assert res.critical_depth == -1.0

    devices[0].coupling_map = ["wrong_coupling_map"]
    res = create_qiskit_result(QuantumCircuit(10), devices[0])
    assert isinstance(res, Result)
    assert res.compilation_time == -1.0
    assert res.fidelity == -1.0
    assert res.critical_depth == -1.0

    res = create_tket_result(QuantumCircuit(10), devices[0])
    assert isinstance(res, Result)
    assert res.compilation_time == -1.0
    assert res.fidelity == -1.0
    assert res.critical_depth == -1.0


def test_result_none_input() -> None:
    res = Result("test", 1.0, None, None)
    assert res.compilation_time == 1.0
    assert res.fidelity == -1.0
    assert res.critical_depth == -1.0
