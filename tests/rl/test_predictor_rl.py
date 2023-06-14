import pytest
from mqt.bench import get_benchmark
from mqt.predictor import rl
from qiskit import QuantumCircuit


def test_instantiate_models() -> None:
    predictor = rl.Predictor()
    predictor.train_all_models(
        timesteps=100,
        reward_functions=["fidelity", "critical_depth", "mix", "gate_ratio"],
        test=True,
    )


@pytest.mark.parametrize(
    "opt_objective",
    ["fidelity", "critical_depth", "gate_ratio", "mix"],
)
def test_qcompile(opt_objective: rl.helper.reward_functions) -> None:
    qc = get_benchmark("ghz", 1, 5)
    qc_compiled, compilation_information = rl.qcompile(qc, opt_objective=opt_objective)
    assert isinstance(qc_compiled, QuantumCircuit)
    assert compilation_information is not None


NUM_EVALUATION_FEATURES = 32


def test_evaluate_sample_circuit() -> None:
    qc = get_benchmark("ghz", 1, 5)
    qc.qasm(filename="test_5.qasm")
    predictor = rl.Predictor()
    res = predictor.evaluate_sample_circuit("test_5.qasm")
    assert len(res) == NUM_EVALUATION_FEATURES
