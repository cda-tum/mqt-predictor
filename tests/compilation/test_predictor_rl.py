import os

import pytest
from mqt.bench import get_benchmark
from mqt.predictor import reward, rl
from qiskit import QuantumCircuit

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS,
    reason="Trained models are not yet available. Skipping may be reverted after release with trained models.",
)
@pytest.mark.parametrize(
    "figure_of_merit",
    ["expected_fidelity", "critical_depth"],
)
def test_qcompile_with_pretrained_models(figure_of_merit: reward.figure_of_merit) -> None:
    qc = get_benchmark("ghz", 1, 5)
    qc_compiled, compilation_information = rl.qcompile(
        qc, figure_of_merit=figure_of_merit, device_name="ibm_washington"
    )
    assert isinstance(qc_compiled, QuantumCircuit)
    assert compilation_information is not None


def test_instantiate_models() -> None:
    predictor = rl.Predictor()
    predictor.train_all_models(
        timesteps=20,
        reward_functions=["expected_fidelity", "critical_depth"],
        device_name="ionq_harmony",
        test=True,
    )


@pytest.mark.parametrize(
    "figure_of_merit",
    ["expected_fidelity", "critical_depth"],
)
def test_qcompile_with_newly_trained_models(figure_of_merit: reward.figure_of_merit) -> None:
    qc = get_benchmark("ghz", 1, 5)
    res = rl.qcompile(qc, figure_of_merit=figure_of_merit, device_name="ionq_harmony")
    assert type(res) == tuple
    qc_compiled, compilation_information = res

    assert isinstance(qc_compiled, QuantumCircuit)
    assert compilation_information is not None
