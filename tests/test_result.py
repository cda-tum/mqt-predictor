from mqt.bench import get_benchmark
from mqt.predictor import Result, rl
from qiskit import transpile


def test_result() -> None:
    qc = get_benchmark("ghz", 1, 5)
    for device in rl.helper.get_devices():
        qc_compiled = transpile(qc, basis_gates=device["native_gates"], coupling_map=device["cmap"])
        res = Result(
            "test_circuit", compiler="test", compilation_time=1.0, compiled_qc=qc_compiled, device=device["name"]
        )
        assert res is not None
