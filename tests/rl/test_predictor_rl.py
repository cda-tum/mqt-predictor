from pathlib import Path

import pytest
from mqt.bench import get_benchmark
from qiskit import QuantumCircuit

from mqt.predictor import rl


@pytest.mark.parametrize(
    "opt_objective",
    ["fidelity", "critical_depth", "parallelism"],
)
def test_compile(opt_objective):
    predictor = rl.Predictor()
    qc = get_benchmark("ghz", 1, 5)
    qc_compiled, device = predictor.compile_prediction(qc, opt_objective=opt_objective)
    assert isinstance(qc_compiled, QuantumCircuit)
    assert device is not None and isinstance(device, str)


def test_evaluate_sample_circuit_using():
    qc = get_benchmark("ghz", 1, 5)
    qc.qasm(filename="test.qasm")
    predictor = rl.Predictor()
    res = predictor.evaluate_sample_circuit_using("test.qasm")
    assert len(res) == 22


def test_instantiate_models():
    predictor = rl.Predictor()
    model_name = "test"
    predictor.instantiate_models(
        timesteps=100, fid=True, dep=False, par=False, model_name=model_name
    )
    path = rl.helper.get_path_trained_model() / (model_name + "_fidelity.zip")
    assert Path(path).is_file()
    if Path(path).is_file():
        Path(path).unlink()
