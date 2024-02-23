from __future__ import annotations

import pytest
from qiskit import QuantumCircuit

from mqt.bench import get_benchmark
from mqt.predictor import reward, rl

# only run test when executed on GitHub runner
IN_GITHUB_ACTIONS = False  # os.getenv("GITHUB_ACTIONS") == "true" TODO: reactivate?


@pytest.mark.skipif(not IN_GITHUB_ACTIONS, reason="Only run this test on GitHub runner")
@pytest.mark.parametrize(
    "figure_of_merit",
    ["expected_fidelity", "critical_depth"],
)
def test_qcompile_with_pretrained_models(figure_of_merit: reward.figure_of_merit) -> None:
    qc = get_benchmark("ghz", 1, 3)
    qc_compiled, compilation_information = rl.qcompile(qc, figure_of_merit=figure_of_merit, device_name="quantinuum_h2")
    assert isinstance(qc_compiled, QuantumCircuit)
    assert compilation_information is not None
