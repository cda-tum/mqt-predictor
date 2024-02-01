from __future__ import annotations

from bqskit.ext import bqskit_to_qiskit, qiskit_to_bqskit
from qiskit import QuantumCircuit

from mqt.predictor.rl import helper


def test_BQSKitO2_action() -> None:
    """Test the BQSKitO2 action."""
    action_BQSKitO2 = None
    for action in helper.get_actions_opt_before_layout():
        if action["name"] == "BQSKitO2":
            action_BQSKitO2 = action

    assert action_BQSKitO2 is not None

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    bqskit_qc = qiskit_to_bqskit(qc)
    altered_qc = bqskit_to_qiskit(action_BQSKitO2["transpile_pass"](bqskit_qc))

    assert altered_qc != qc
