from mqt.bench import get_one_benchmark
from pytket import Circuit
from qiskit import QuantumCircuit

from mqt.predictor.driver import compile


def test_compile():
    qc = get_one_benchmark("ghz", 1, 5)
    for model in ["ML"]:
        qc_compiled, device = compile(qc, model=model)
        assert isinstance(qc_compiled, (QuantumCircuit, Circuit))
        assert isinstance(device, str)
