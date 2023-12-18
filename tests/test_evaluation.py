from __future__ import annotations

from pathlib import Path

from mqt.bench import get_benchmark
from mqt.bench.devices import get_available_devices
from mqt.predictor.evaluation import evaluate_sample_circuit


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
                    [compilation_setup + "_" + device.name + "_" + key for device in get_available_devices()]
                )

    assert all(key in res for key in expected_keys)
    if Path(filename).exists():
        Path(filename).unlink()
