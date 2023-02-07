from pathlib import Path

import pytest
from mqt.bench import get_benchmark
from mqt.predictor.rl import Predictor, RewardFunction, qcompile
from mqt.predictor.rl.helper import get_path_trained_model
from qiskit import QuantumCircuit


@pytest.mark.parametrize(
    "opt_objective",
    ["fidelity", "critical_depth", "gate_ratio", "mix"],
)
def test_qcompile(opt_objective: RewardFunction) -> None:
    qc = get_benchmark("ghz", 1, 5)
    qc_compiled, compilation_information = qcompile(qc, opt_objective=opt_objective)
    assert isinstance(qc_compiled, QuantumCircuit)
    assert compilation_information is not None


NUM_EVALUATION_FEATURES = 32


def test_evaluate_sample_circuit() -> None:
    qc = get_benchmark("ghz", 1, 5)
    qc.qasm(filename="test_5.qasm")
    predictor = Predictor()
    res = predictor.evaluate_sample_circuit(Path("test_5.qasm"))
    assert len(res) == NUM_EVALUATION_FEATURES


def test_instantiate_models() -> None:
    predictor = Predictor()
    predictor.train_all_models(
        timesteps=100,
        reward_functions=["fidelity", "critical_depth", "mix", "gate_ratio"],
        model_name="test",
    )
    path_fid = get_path_trained_model() / "test_fidelity.zip"
    path_dep = get_path_trained_model() / "test_critical_depth.zip"
    path_gates = get_path_trained_model() / "test_gate_ratio.zip"
    path_mix = get_path_trained_model() / "test_mix.zip"

    paths = [path_fid, path_dep, path_gates, path_mix]
    for path in paths:
        assert Path(path).is_file()
        if Path(path).is_file():
            Path(path).unlink()
