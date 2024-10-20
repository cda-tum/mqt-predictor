"""Tests for the compilation with reinforcement learning."""

from __future__ import annotations

from pathlib import Path

import pytest
from qiskit import QuantumCircuit
from qiskit.qasm2 import dump

from mqt.bench import get_benchmark
from mqt.predictor import reward, rl


def test_predictor_env_reset_from_string() -> None:
    """Test the reset function of the predictor environment with a quantum circuit given as a string as input."""
    predictor = rl.Predictor(figure_of_merit="expected_fidelity", device_name="ionq_harmony")
    qasm_path = Path("test.qasm")
    qc = get_benchmark("dj", 1, 3)
    with qasm_path.open("w", encoding="utf-8") as f:
        dump(qc, f)
    assert predictor.env.reset(qc=qasm_path)[0] == rl.helper.create_feature_dict(qc)


@pytest.mark.parametrize(
    "figure_of_merit",
    ["expected_fidelity", "critical_depth"],
)
def test_qcompile_with_newly_trained_models(figure_of_merit: reward.figure_of_merit) -> None:
    """Test the qcompile function with a newly trained model."""
    """ Important: Those trained models are used in later tests and must not be deleted. """

    device = "ibm_montreal"
    qc = get_benchmark("ghz", 1, 5)
    predictor = rl.Predictor(figure_of_merit=figure_of_merit, device_name=device)

    model_name = "model_" + figure_of_merit + "_" + device
    if not Path(rl.helper.get_path_trained_model() / (model_name + ".zip")).exists():
        with pytest.raises(
            FileNotFoundError, match="The RL model is not trained yet. Please train the model before using it."
        ):
            rl.qcompile(qc, figure_of_merit=figure_of_merit, device_name=device)

    predictor.train_model(
        timesteps=100,
        test=True,
    )

    res = rl.qcompile(qc, figure_of_merit=figure_of_merit, device_name=device)
    assert isinstance(res, tuple)
    qc_compiled, compilation_information = res

    assert isinstance(qc_compiled, QuantumCircuit)
    assert qc_compiled.layout is not None
    assert compilation_information is not None


def test_qcompile_with_false_input() -> None:
    """Test the qcompile function with false input."""
    qc = get_benchmark("dj", 1, 5)
    with pytest.raises(ValueError, match="figure_of_merit must not be None if predictor_singleton is None."):
        rl.helper.qcompile(qc, None, "quantinuum_h2")
    with pytest.raises(ValueError, match="device_name must not be None if predictor_singleton is None."):
        rl.helper.qcompile(qc, "expected_fidelity", None)
