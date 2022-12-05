import os
from pathlib import Path

import pytest
from mqt.bench import get_benchmark
from qiskit import QuantumCircuit

from mqt.predictor import RL_utils
from mqt.predictor.RL_Predictor import RL_Predictor


@pytest.mark.parametrize(
    "opt_objective",
    ["fidelity", "critical_depth", "parallelism"],
)
def test_compile(opt_objective):
    predictor = RL_Predictor()
    qc = get_benchmark("ghz", 1, 5)
    qc_compiled, device = predictor.compile(qc, opt_objective=opt_objective)
    assert isinstance(qc_compiled, QuantumCircuit)
    assert device is not None and isinstance(device, str)


def test_evaluate_sample_circuit_using_RL():
    qc = get_benchmark("ghz", 1, 5)
    qc.qasm(filename="test.qasm")
    predictor = RL_Predictor()
    res = predictor.evaluate_sample_circuit_using_RL("test.qasm")
    assert len(res) == 22


@pytest.mark.skipif(
    os.getenv("skip_optional_tests"), reason="Takes too long on GitHub Runner."
)
def test_instantiate_RL_models():
    predictor = RL_Predictor()
    model_name = "test"
    predictor.instantiate_RL_models(
        1, fid=True, dep=False, par=False, model_name=model_name
    )
    path = RL_utils.get_path_trained_model_RL() / (model_name + "_fidelity.zip")
    assert Path(path).is_file()
    if Path(path).is_file():
        Path(path).unlink()
