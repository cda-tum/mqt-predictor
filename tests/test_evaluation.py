"""Tests for the evaluation module."""

from __future__ import annotations

import copy
import io
import logging

from qiskit import QuantumCircuit

from mqt.bench.devices import get_device_by_name
from mqt.predictor import Result
from mqt.predictor.evaluation import create_qiskit_result, create_tket_result
from mqt.predictor.reward import esp_data_available


def test_create_result() -> None:
    """Test the creation of a result object."""
    device = get_device_by_name("iqm_apollo")
    assert device.num_qubits >= 10
    qc = QuantumCircuit(10)
    qc.measure_all()

    res = create_tket_result(qc, device)
    assert isinstance(res, Result)
    assert res.compilation_time >= 0.0
    assert res.expected_fidelity >= 0.0
    assert res.critical_depth >= 0.0
    assert res.estimated_success_probability >= 0.0
    assert res.estimated_hellinger_distance >= 0.0

    res = create_qiskit_result(qc, device)
    assert isinstance(res, Result)
    assert res.compilation_time >= 0.0
    assert res.expected_fidelity >= 0.0
    assert res.critical_depth >= 0.0
    assert res.estimated_success_probability >= 0.0
    assert res.estimated_hellinger_distance >= 0.0


def test_false_input() -> None:
    """Test the creation of a result object with false input."""
    device = get_device_by_name("ionq_harmony")

    res = create_tket_result(QuantumCircuit(1000), device)
    assert isinstance(res, Result)
    assert res.compilation_time == -1.0
    assert res.expected_fidelity == -1.0
    assert res.critical_depth == -1.0
    assert res.estimated_success_probability == -1.0
    assert res.estimated_hellinger_distance == -1.0

    res = create_qiskit_result(QuantumCircuit(1000), device)
    assert isinstance(res, Result)
    assert res.compilation_time == -1.0
    assert res.expected_fidelity == -1.0
    assert res.critical_depth == -1.0
    assert res.estimated_success_probability == -1.0
    assert res.estimated_hellinger_distance == -1.0

    device.coupling_map = ["wrong_coupling_map"]
    res = create_qiskit_result(QuantumCircuit(10), device)
    assert isinstance(res, Result)
    assert res.compilation_time == -1.0
    assert res.expected_fidelity == -1.0
    assert res.critical_depth == -1.0
    assert res.estimated_success_probability == -1.0
    assert res.estimated_hellinger_distance == -1.0

    res = create_tket_result(QuantumCircuit(10), device)
    assert isinstance(res, Result)
    assert res.compilation_time == -1.0
    assert res.expected_fidelity == -1.0
    assert res.critical_depth == -1.0
    assert res.estimated_success_probability == -1.0
    assert res.estimated_hellinger_distance == -1.0


def test_result_none_input() -> None:
    """Test the creation of a result object with None input."""
    res = Result("test", 1.0, None, None)
    assert res.compilation_time == 1.0
    assert res.expected_fidelity == -1.0
    assert res.critical_depth == -1.0
    assert res.estimated_success_probability == -1.0
    assert res.estimated_hellinger_distance == -1.0


def test_esp_data_available() -> None:
    """Test the ESP calibration data check."""
    device = get_device_by_name("ionq_harmony")

    # Set up logging to capture log messages for verification
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)
    log_capture_string = io.StringIO()
    ch = logging.StreamHandler(log_capture_string)
    ch.setLevel(logging.ERROR)
    logger.addHandler(ch)

    # Test missing T1 data
    error_device = copy.deepcopy(device)
    error_device.calibration.t1 = None
    assert not esp_data_available(error_device)
    log_contents = log_capture_string.getvalue()
    assert (
        "T1 data for idle operation on qubit(s) 0 is required to calculate ESP for device ionq_harmony." in log_contents
    )

    # Test missing T2 data
    log_capture_string.truncate(0)
    log_capture_string.seek(0)
    error_device = copy.deepcopy(device)
    error_device.calibration.t2 = None
    assert not esp_data_available(error_device)
    log_contents = log_capture_string.getvalue()
    assert (
        "T2 data for idle operation on qubit(s) 0 is required to calculate ESP for device ionq_harmony." in log_contents
    )

    # Test missing readout fidelity
    log_capture_string.truncate(0)
    log_capture_string.seek(0)
    error_device = copy.deepcopy(device)
    error_device.calibration.readout_fidelity = None
    assert not esp_data_available(error_device)
    log_contents = log_capture_string.getvalue()
    assert (
        "Fidelity data for readout operation on qubit(s) 0 is required to calculate ESP for device ionq_harmony."
        in log_contents
    )

    # Test missing readout duration
    log_capture_string.truncate(0)
    log_capture_string.seek(0)
    error_device = copy.deepcopy(device)
    error_device.calibration.readout_duration = None
    assert not esp_data_available(error_device)
    log_contents = log_capture_string.getvalue()
    assert (
        "Duration data for readout operation on qubit(s) 0 is required to calculate ESP for device ionq_harmony."
        in log_contents
    )

    # Define erroneous gate and qubit
    error_gate = device.get_single_qubit_gates().pop()
    error_qubit = 0

    # Test missing single qubit gate fidelity
    log_capture_string.truncate(0)
    log_capture_string.seek(0)
    error_device = copy.deepcopy(device)
    error_device.calibration.single_qubit_gate_fidelity[error_qubit].pop(error_gate)
    assert not esp_data_available(error_device)
    log_contents = log_capture_string.getvalue()
    assert (
        f"Fidelity data for {error_gate} operation on qubit(s) {error_qubit} is required to calculate ESP for device ionq_harmony."
        in log_contents
    )

    # Test missing single qubit gate duration
    log_capture_string.truncate(0)
    log_capture_string.seek(0)
    error_device = copy.deepcopy(device)
    error_device.calibration.single_qubit_gate_duration[error_qubit].pop(error_gate)
    assert not esp_data_available(error_device)
    log_contents = log_capture_string.getvalue()
    assert (
        f"Duration data for {error_gate} operation on qubit(s) {error_qubit} is required to calculate ESP for device ionq_harmony."
        in log_contents
    )

    # Define erroneous two-qubit gate and edge
    error_gate = device.get_two_qubit_gates().pop()
    error_edge = device.coupling_map[0]

    # Test missing two qubit gate fidelity
    log_capture_string.truncate(0)
    log_capture_string.seek(0)
    error_device = copy.deepcopy(device)
    error_device.calibration.two_qubit_gate_fidelity[tuple(error_edge)].pop(error_gate)
    assert not esp_data_available(error_device)
    log_contents = log_capture_string.getvalue()
    assert (
        f"Fidelity data for {error_gate} operation on qubit(s) {error_edge} is required to calculate ESP for device ionq_harmony."
        in log_contents
    )

    # Test missing two qubit gate duration
    log_capture_string.truncate(0)
    log_capture_string.seek(0)
    error_device = copy.deepcopy(device)
    error_device.calibration.two_qubit_gate_duration[tuple(error_edge)].pop(error_gate)
    assert not esp_data_available(error_device)
    log_contents = log_capture_string.getvalue()
    assert (
        f"Duration data for {error_gate} operation on qubit(s) {error_edge} is required to calculate ESP for device ionq_harmony."
        in log_contents
    )

    # Finally, assert that all required data is available in the original device
    assert esp_data_available(device)

    # Clean up logging
    logger.removeHandler(ch)
    log_capture_string.close()
