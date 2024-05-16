"""Tests for the evaluation module."""

from __future__ import annotations

from qiskit import QuantumCircuit

from mqt.bench.devices import get_device_by_name
from mqt.predictor import Result
from mqt.predictor.evaluation import create_qiskit_result, create_tket_result


def test_create_result() -> None:
    """Test the creation of a result object."""
    device = get_device_by_name("ibm_montreal")
    assert device.num_qubits >= 10
    qc = QuantumCircuit(10)
    qc.measure_all()

    res = create_tket_result(qc, device)
    assert isinstance(res, Result)
    assert res.compilation_time >= 0.0
    assert res.expected_fidelity >= 0.0
    assert res.critical_depth >= 0.0
    assert res.expected_success_probability >= 0.0

    res = create_qiskit_result(qc, device)
    assert isinstance(res, Result)
    assert res.compilation_time >= 0.0
    assert res.expected_fidelity >= 0.0
    assert res.critical_depth >= 0.0
    assert res.expected_success_probability >= 0.0


def test_false_input() -> None:
    """Test the creation of a result object with false input."""
    device = get_device_by_name("ibm_montreal")

    res = create_tket_result(QuantumCircuit(1000), device)
    assert isinstance(res, Result)
    assert res.compilation_time == -1.0
    assert res.expected_fidelity == -1.0
    assert res.critical_depth == -1.0
    assert res.expected_success_probability == -1.0

    res = create_qiskit_result(QuantumCircuit(1000), device)
    assert isinstance(res, Result)
    assert res.compilation_time == -1.0
    assert res.expected_fidelity == -1.0
    assert res.critical_depth == -1.0
    assert res.expected_success_probability == -1.0

    device.coupling_map = ["wrong_coupling_map"]
    res = create_qiskit_result(QuantumCircuit(10), device)
    assert isinstance(res, Result)
    assert res.compilation_time == -1.0
    assert res.expected_fidelity == -1.0
    assert res.critical_depth == -1.0
    assert res.expected_success_probability == -1.0

    res = create_tket_result(QuantumCircuit(10), device)
    assert isinstance(res, Result)
    assert res.compilation_time == -1.0
    assert res.expected_fidelity == -1.0
    assert res.critical_depth == -1.0
    assert res.expected_success_probability == -1.0


def test_result_none_input() -> None:
    """Test the creation of a result object with None input."""
    res = Result("test", 1.0, None, None)
    assert res.compilation_time == 1.0
    assert res.expected_fidelity == -1.0
    assert res.critical_depth == -1.0
    assert res.expected_success_probability == -1.0
