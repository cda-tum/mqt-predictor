from __future__ import annotations

from qiskit import QuantumCircuit

from mqt.bench import get_benchmark
from mqt.bench.devices import get_available_device_names, get_available_devices
from mqt.predictor import Result
from mqt.predictor.evaluation import create_qiskit_result, create_tket_result, evaluate_sample_circuit


def test_evaluate_sample_circuit() -> None:
    qc = get_benchmark("ghz", 1, 3)
    filename = "test_3.qasm"
    qc.qasm(filename=filename)
    res = evaluate_sample_circuit(filename, get_available_devices())
    expected_keys = []
    for compilation_setup in ["qiskit", "tket", "mqt-predictor_expected_fidelity", "mqt-predictor_critical_depth"]:
        for key in ["time", "expected_fidelity", "critical_depth"]:
            if "mqt-predictor" in compilation_setup:
                expected_keys.append(compilation_setup + "_" + key)
            else:
                expected_keys.extend(
                    [compilation_setup + "_" + device_name + "_" + key for device_name in get_available_device_names()]
                )

    res = create_qiskit_result(QuantumCircuit(10), devices[0])
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
