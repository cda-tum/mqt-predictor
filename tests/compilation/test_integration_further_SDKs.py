from __future__ import annotations

from typing import Any, cast

import pytest
from bqskit.ext import bqskit_to_qiskit, qiskit_to_bqskit
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import CheckMap, GatesInBasis
from qiskit.transpiler.runningpassmanager import TranspileLayout

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
    optimized_qc = bqskit_to_qiskit(action_BQSKitO2["transpile_pass"](bqskit_qc))

    assert optimized_qc != qc


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
    native_gates_qc = bqskit_to_qiskit(transpile_pass(bqskit_qc))

    check_nat_gates = GatesInBasis(basis_gates=device["native_gates"])
    check_nat_gates(native_gates_qc)
    only_nat_gates = check_nat_gates.property_set["all_gates_in_basis"]
    if "oqc" not in device["name"]:
        assert only_nat_gates


def test_BQSKitMapping_action_swaps_necessary() -> None:
    """Test the BQSKitMapping action for quantum circuit that requires SWAP gates."""
    action_BQSKitMapping = None
    for action in helper.get_actions_mapping():
        if action["name"] == "BQSKitMapping":
            action_BQSKitMapping = action

    assert action_BQSKitMapping is not None

    qc = QuantumCircuit(5)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.cx(0, 3)
    qc.cx(0, 4)

    device = helper.get_devices()[1]
    bqskit_qc = qiskit_to_bqskit(qc)
    bqskit_qc_mapped, input_mapping, output_mapping = action_BQSKitMapping["transpile_pass"](device)(bqskit_qc)
    mapped_qc = bqskit_to_qiskit(bqskit_qc_mapped)
    layout = helper.final_layout_bqskit_to_qiskit(input_mapping, output_mapping, mapped_qc, qc)

    assert input_mapping != output_mapping
    assert layout.final_layout is not None
    check_mapped_circuit(initial_qc=qc, mapped_qc=mapped_qc, device=device, layout=layout)


def check_mapped_circuit(
    initial_qc: QuantumCircuit, mapped_qc: QuantumCircuit, device: dict[str, Any], layout: TranspileLayout
) -> None:
    # check if the altered circuit is correctly mapped to the device
    check_mapping = CheckMap(coupling_map=CouplingMap(device["cmap"]))
    check_mapping(mapped_qc)
    mapped = check_mapping.property_set["is_swap_mapped"]
    assert mapped
    assert mapped_qc != initial_qc
    assert layout is not None
    assert len(layout.initial_layout) == device["max_qubits"]
    if layout.final_layout is not None:
        assert len(layout.final_layout) == device["max_qubits"]

    # each qubit of the initial layout is part of the initial quantum circuit and the register name is correctly set
    for assigned_physical_qubit in layout.initial_layout._p2v.values():  # noqa: SLF001
        qreg = assigned_physical_qubit.register
        assert qreg.name in {"q", "ancilla"}

        # assigned_physical_qubit is part of the original quantum circuit
        if qreg.name == "q":
            assert qreg.size == initial_qc.num_qubits
            # each qubit is also part of the initial uncompiled quantum circuit
            assert initial_qc.find_bit(assigned_physical_qubit).registers[0][0].name == "q"
        # assigned_physical_qubit is an ancilla qubit
        else:
            assert qreg.size == device["max_qubits"] - initial_qc.num_qubits
    # each qubit of the final layout is part of the mapped quantum circuit and the register name is correctly set
    if layout.final_layout is not None:
        for assigned_physical_qubit in layout.final_layout._p2v.values():  # noqa: SLF001
            assert mapped_qc.find_bit(assigned_physical_qubit).registers[0][0].name == "q"
    # each virtual qubit of the original quantum circuit is part of the initial layout
    for virtual_qubit in initial_qc.qubits:
        assert virtual_qubit in layout.initial_layout._p2v.values()  # noqa: SLF001


def test_BQSKitMapping_action_no_swaps_necessary() -> None:
    """Test the BQSKitMapping action for a simple quantum circuit that does not require SWAP gates."""

    action_BQSKitMapping = None
    for action in helper.get_actions_mapping():
        if action["name"] == "BQSKitMapping":
            action_BQSKitMapping = action

    assert action_BQSKitMapping is not None

    qc_no_swap_needed = QuantumCircuit(2)
    qc_no_swap_needed.h(0)
    qc_no_swap_needed.cx(0, 1)

    device = helper.get_devices()[1]

    bqskit_qc = qiskit_to_bqskit(qc_no_swap_needed)
    bqskit_qc_mapped, input_mapping, output_mapping = action_BQSKitMapping["transpile_pass"](device)(bqskit_qc)
    mapped_qc = bqskit_to_qiskit(bqskit_qc_mapped)
    layout = helper.final_layout_bqskit_to_qiskit(input_mapping, output_mapping, mapped_qc, qc_no_swap_needed)
    assert layout is not None
    assert input_mapping == output_mapping
    assert layout.final_layout is None

    check_mapped_circuit(qc_no_swap_needed, mapped_qc, device, layout)


def test_TKET_routing() -> None:
    """Test the TKETRouting action."""

    qc = QuantumCircuit(5)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.cx(0, 3)
    qc.cx(0, 4)

    device = helper.get_devices()[1]

    layout_action = helper.get_actions_layout()[0]
    transpile_pass = layout_action["transpile_pass"](device)
    pm = PassManager(transpile_pass)
    layouted_qc = pm.run(qc)
    initial_layout = pm.property_set["layout"]
    input_qubit_mapping = pm.property_set["original_qubit_indices"]

    routing_action = None
    for action in helper.get_actions_routing():
        if action["origin"] == "tket":
            routing_action = action
    assert routing_action is not None

    tket_qc = qiskit_to_tk(layouted_qc, preserve_param_uuid=True)
    for elem in routing_action["transpile_pass"](device):
        elem.apply(tket_qc)
    from pytket.circuit import Qubit

    qbs = tket_qc.qubits
    qubit_map = {qbs[i]: Qubit("q", i) for i in range(len(qbs))}
    tket_qc.rename_units(qubit_map)  # type: ignore[arg-type]

    mapped_qc = tk_to_qiskit(tket_qc)

    final_layout = helper.final_layout_pytket_to_qiskit(tket_qc, mapped_qc)

    layout = TranspileLayout(
        initial_layout=initial_layout,
        input_qubit_mapping=input_qubit_mapping,
        final_layout=final_layout,
        _output_qubit_list=mapped_qc.qubits,
        _input_qubit_count=qc.num_qubits,
    )

    check_mapped_circuit(qc, mapped_qc, device, layout)