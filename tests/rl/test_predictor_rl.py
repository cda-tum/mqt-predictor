from pathlib import Path

import pytest
from mqt.bench import get_benchmark
from qiskit import QuantumCircuit

from mqt.predictor import rl


@pytest.mark.parametrize(
    "opt_objective",
    ["fidelity", "critical_depth", "gate_ratio", "mix"],
)
def test_qcompile(opt_objective):
    qc = get_benchmark("ghz", 1, 5)
    qc_compiled, compilation_information = rl.qcompile(qc, opt_objective=opt_objective)
    assert isinstance(qc_compiled, QuantumCircuit)
    assert compilation_information is not None


def test_evaluate_sample_circuit():
    qc = get_benchmark("ghz", 1, 5)
    qc.qasm(filename="test_5.qasm")
    predictor = rl.Predictor()
    res = predictor.evaluate_sample_circuit("test_5.qasm")
    assert len(res) == 32


def test_instantiate_models():
    predictor = rl.Predictor()
    predictor.train_all_models(
        timesteps=100,
        reward_functions=["fidelity", "critical_depth", "mix", "gate_ratio"],
        model_name="test",
    )
    path_fid = rl.helper.get_path_trained_model() / "test_fidelity.zip"
    path_dep = rl.helper.get_path_trained_model() / "test_critical_depth.zip"
    path_gates = rl.helper.get_path_trained_model() / "test_gate_ratio.zip"
    path_mix = rl.helper.get_path_trained_model() / "test_mix.zip"

    paths = [path_fid, path_dep, path_gates, path_mix]
    for path in paths:
        assert Path(path).is_file()
        if Path(path).is_file():
            Path(path).unlink()
