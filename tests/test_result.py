from mqt.predictor import Result
from mqt.bench import get_benchmark
from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import FakeMontreal, FakeWashington
from mqt.predictor import rl

def test_result() -> None:
    qc = get_benchmark("ghz", 1, 5)
    for device in rl.helper.get_devices():
        qc_compiled = transpile(qc, basis_gates=device["native_gates"], coupling_map=device["cmap"])
        assert Result("test_circuit", used_setup="test", duration=1.0, qc = qc_compiled, device=device["name"])