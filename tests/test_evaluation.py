from pathlib import Path

from mqt.bench import get_benchmark
from mqt.predictor.evaluation import evaluate_sample_circuit


def test_evaluate_sample_circuit() -> None:
    qc = get_benchmark("ghz", 1, 3)
    filename = "test_3.qasm"
    qc.qasm(filename=filename)
    res = evaluate_sample_circuit(filename)
    expected_keys = [
        "file_path",
        "benchmark_name",
        "num_qubits",
        "mqt-predictor_expected_fidelity_expected_fidelity",
        "mqt-predictor_critical_depth_critical_depth",
    ]

    assert all(key in res for key in expected_keys)
    if Path(filename).exists():
        Path(filename).unlink()
