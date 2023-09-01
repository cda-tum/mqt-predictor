import pytest
from mqt.bench import get_benchmark
from mqt.predictor import reward, rl
from qiskit import QuantumCircuit


@pytest.mark.parametrize(
    "figure_of_merit",
    ["fidelity", "critical_depth"],
)
def test_qcompile_with_pretrained_models(figure_of_merit: reward.reward_functions) -> None:
    qc = get_benchmark("ghz", 1, 5)
    res = rl.qcompile(qc, figure_of_merit=figure_of_merit)
    assert type(res) == tuple
    qc_compiled, compilation_information = res
    assert isinstance(qc_compiled, QuantumCircuit)
    assert compilation_information is not None


def test_instantiate_models() -> None:
    predictor = rl.Predictor()
    predictor.train_all_models(
        timesteps=100,
        reward_functions=["fidelity", "critical_depth"],
        test=True,
    )


@pytest.mark.parametrize(
    "figure_of_merit",
    ["fidelity", "critical_depth"],
)
def test_qcompile_with_newly_trained_models(figure_of_merit: reward.reward_functions) -> None:
    qc = get_benchmark("ghz", 1, 5)
    res = rl.qcompile(qc, figure_of_merit=figure_of_merit)
    assert type(res) == tuple
    qc_compiled, compilation_information = res

    assert isinstance(qc_compiled, QuantumCircuit)
    assert compilation_information is not None
