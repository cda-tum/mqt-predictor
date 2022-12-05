from mqt.predictor.driver import compile
from mqt.bench import get_benchmark
from qiskit import QuantumCircuit
from pytket import Circuit

def test_compile():
    qc = get_benchmark("ghz", 1, 5)
    for model in ["ML", "RL"]:
        qc_compiled, device = compile(qc, model=model)
        assert isinstance(qc_compiled, QuantumCircuit | Circuit)
        assert isinstance(device, str)
