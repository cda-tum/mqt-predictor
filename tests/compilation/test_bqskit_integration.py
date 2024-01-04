from mqt.predictor.rl import helper
from qiskit import QuantumCircuit
from bqskit.ext import bqskit_to_qiskit, qiskit_to_bqskit
import pytest
from typing import Any, cast
from qiskit.transpiler.passes import GatesInBasis

def test_BQSKitO2_action() -> None:
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

@pytest.mark.parametrize(
    "device",
    helper.get_devices(),
    ids=lambda device: cast(str, device["name"])
)
def test_BQSKitSynthesis_action(device: dict[str, Any]) -> None:
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
    assert check_nat_gates.property_set["all_gates_in_basis"]

    transpile_pass = action_BQSKitSynthesis["transpile_pass"](qc.num_qubits, device["name"].split("_")[0])
    bqskit_qc = qiskit_to_bqskit(qc)
    altered_qc = bqskit_to_qiskit(transpile_pass(bqskit_qc))

    check_nat_gates = GatesInBasis(basis_gates=device["native_gates"])
    check_nat_gates(altered_qc)
    only_nat_gates = check_nat_gates.property_set["all_gates_in_basis"]
    if "oqc" not in device["name"]:
        assert only_nat_gates




