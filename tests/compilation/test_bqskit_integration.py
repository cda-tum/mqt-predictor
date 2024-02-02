from __future__ import annotations

from typing import Any, cast

import pytest
from bqskit.ext import bqskit_to_qiskit, qiskit_to_bqskit
from qiskit import QuantumCircuit
from qiskit.transpiler.passes import GatesInBasis

from mqt.predictor.rl import helper


def test_BQSKitO2_action() -> None:
    """Test the BQSKitO2 action."""
    action_BQSKitO2 = None
    for action in helper.get_actions_opt():
        if action["name"] == "BQSKitO2":
            action_BQSKitO2 = action

    assert action_BQSKitO2 is not None

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    bqskit_qc = qiskit_to_bqskit(qc)
    altered_qc = bqskit_to_qiskit(action_BQSKitO2["transpile_pass"](bqskit_qc))

    assert altered_qc != qc


@pytest.mark.parametrize("device", helper.get_devices(), ids=lambda device: cast(str, device["name"]))
def test_BQSKitSynthesis_action(device: dict[str, Any]) -> None:
    """Test the BQSKitSynthesis action for all devices."""
    action_BQSKitSynthesis = None
    for action in helper.get_actions_synthesis():
        if action["name"] == "BQSKitSynthesis":
            action_BQSKitSynthesis = action

    assert action_BQSKitSynthesis is not None

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    check_nat_gates = GatesInBasis(basis_gates=device["native_gates"])
    check_nat_gates(qc)
    assert not check_nat_gates.property_set["all_gates_in_basis"]

    transpile_pass = action_BQSKitSynthesis["transpile_pass"](device)
    bqskit_qc = qiskit_to_bqskit(qc)
    altered_qc = bqskit_to_qiskit(transpile_pass(bqskit_qc))

    check_nat_gates = GatesInBasis(basis_gates=device["native_gates"])
    check_nat_gates(altered_qc)
    only_nat_gates = check_nat_gates.property_set["all_gates_in_basis"]
    if "oqc" not in device["name"]:
        assert only_nat_gates


def test_BQSKitMapping_action() -> None:
    """Test the BQSKitMapping action."""
    action_BQSKitMapping = None
    for action in helper.get_actions_mapping():
        if action["name"] == "BQSKitMapping":
            action_BQSKitMapping = action

    assert action_BQSKitMapping is not None

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    device = helper.get_devices()[1]
    bqskit_qc = qiskit_to_bqskit(qc)
    bqskit_qc_mapped, input_mapping, output_mapping = action_BQSKitMapping["transpile_pass"](device)(bqskit_qc)
    altered_qc = bqskit_to_qiskit(bqskit_qc_mapped)
    layout = helper.final_layout_bqskit_to_qiskit(input_mapping, output_mapping, altered_qc, qc)

    assert altered_qc != qc
    assert layout is not None
