from mqt.bench import get_benchmark
from pytket import Circuit
from qiskit import QuantumCircuit

from mqt.predictor.driver import compile


def test_compile():
    qc = get_benchmark("ghz", 1, 5)
    for model in ["ML", "RL"]:
        qc_compiled, compilation_information = compile(qc, model=model)
        print(qc_compiled, compilation_information)
        assert isinstance(qc_compiled, (QuantumCircuit, Circuit))
        assert compilation_information is not None
