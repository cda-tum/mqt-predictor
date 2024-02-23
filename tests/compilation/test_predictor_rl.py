from __future__ import annotations

import os
from pathlib import Path

import pytest
from qiskit import QuantumCircuit

from mqt.bench import get_benchmark
from mqt.predictor import reward, rl

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.parametrize(
    "figure_of_merit",
    ["expected_fidelity", "critical_depth"],
)
def test_qcompile_with_pretrained_models(figure_of_merit: reward.figure_of_merit) -> None:
    qc = get_benchmark("ghz", 1, 3)
    qc_compiled, compilation_information = rl.qcompile(qc, figure_of_merit=figure_of_merit, device_name="ionq_harmony")
    assert isinstance(qc_compiled, QuantumCircuit)
    assert compilation_information is not None


def test_predictor_env_reset_from_string() -> None:
    predictor = rl.Predictor(figure_of_merit="expected_fidelity", device_name="ionq_harmony")
    qasm_path = Path("test.qasm")
    qc = get_benchmark("dj", 1, 3)
    qc.qasm(filename=str(qasm_path))
    assert predictor.env.reset(qc=qasm_path)[0] == rl.helper.create_feature_dict(qc)


@pytest.mark.parametrize(
    "figure_of_merit",
    ["expected_fidelity", "critical_depth"],
)
def test_qcompile_with_newly_trained_models(figure_of_merit: reward.figure_of_merit) -> None:
    predictor = rl.Predictor(figure_of_merit=figure_of_merit, device_name="ionq_harmony")
    predictor.train_model(
        timesteps=100,
        test=True,
    )

    qc = get_benchmark("ghz", 1, 5)
    res = rl.qcompile(qc, figure_of_merit=figure_of_merit, device_name="ionq_harmony")
    assert type(res) == tuple
    qc_compiled, compilation_information = res

    assert isinstance(qc_compiled, QuantumCircuit)
    assert compilation_information is not None


def test_qcompile_with_false_input() -> None:
    qc = get_benchmark("dj", 1, 5)
    with pytest.raises(ValueError, match="figure_of_merit must not be None if predictor_singleton is None."):
        rl.helper.qcompile(qc, None, "quantinuum_h2")
    with pytest.raises(ValueError, match="device_name must not be None if predictor_singleton is None."):
        rl.helper.qcompile(qc, "expected_fidelity", None)
