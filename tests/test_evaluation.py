from pathlib import Path

from mqt.bench import get_benchmark
from mqt.predictor.evaluation import evaluate_sample_circuit


def test_evaluate_sample_circuit() -> None:
    qc = get_benchmark("ghz", 1, 5)
    filename = "test_5.qasm"
    qc.qasm(filename=filename)
    res = evaluate_sample_circuit(filename)
    assert res
    if Path(filename).exists():
        Path(filename).unlink()
