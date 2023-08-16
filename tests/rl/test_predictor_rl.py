import pytest
from mqt.bench import get_benchmark
from mqt.predictor import rl
from qiskit import QuantumCircuit


@pytest.mark.parametrize(
    "opt_objective",
    ["fidelity", "critical_depth", "gate_ratio", "mix"],
)
def test_qcompile_with_pretrained_models(opt_objective: rl.helper.reward_functions) -> None:
    qc = get_benchmark("ghz", 1, 5)
    res = rl.qcompile(qc, opt_objective=opt_objective)
    assert type(res) == tuple
    qc_compiled, compilation_information = res
    assert isinstance(qc_compiled, QuantumCircuit)
    assert compilation_information is not None


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
def test_qcompile_with_newly_trained_models(opt_objective: rl.helper.reward_functions) -> None:
    qc = get_benchmark("ghz", 1, 5)
    res = rl.qcompile(qc, opt_objective=opt_objective)
    assert type(res) == tuple
    qc_compiled, compilation_information = res

    assert isinstance(qc_compiled, QuantumCircuit)
    assert compilation_information is not None
