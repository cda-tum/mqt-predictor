from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pathlib import Path

    from mqt.predictor.devices import Device
    from qiskit.circuit import AncillaQubit, Qubit

import numpy as np
from qiskit import AncillaRegister, QuantumCircuit, QuantumRegister
from qiskit.transpiler.passes import RemoveBarriers

logger = logging.getLogger("mqtpredictor")


def calc_qubit_index(
    qargs: list[Qubit | AncillaQubit], qregs: list[QuantumRegister | AncillaRegister], index: int
) -> int:
    offset = 0
    for reg in qregs:
        if qargs[index] in reg:
            return offset + cast(int, reg.index(qargs[index]))
        offset += reg.size
    error_msg = "Qubit " + str(qargs[index]) + " not found in any register."
    raise ValueError(error_msg)


def calc_supermarq_features(
    qc: QuantumCircuit,
) -> tuple[float, float, float, float, float]:
    qc = RemoveBarriers()(qc)
    connectivity_collection: list[list[int]] = [[] for _ in range(qc.num_qubits)]
    liveness_A_matrix = 0
    for _, qargs, _ in qc.data:
        liveness_A_matrix += len(qargs)
        all_indices = [calc_qubit_index(qargs, qc.qregs, 0)]
        if len(qargs) == 2:  # noqa: PLR2004
            second_qubit = calc_qubit_index(qargs, qc.qregs, 1)
            all_indices.append(second_qubit)
        for qubit_index in all_indices:
            to_be_added_entries = all_indices.copy()
            to_be_added_entries.remove(qubit_index)
            connectivity_collection[qubit_index].extend(to_be_added_entries)

    connectivity = [len(set(connectivity_collection[i])) for i in range(qc.num_qubits)]

    num_gates = sum(qc.count_ops().values())
    num_multiple_qubit_gates = qc.num_nonlocal_gates()
    depth = qc.depth()
    program_communication = np.sum(connectivity) / (qc.num_qubits * (qc.num_qubits - 1))

    if num_multiple_qubit_gates == 0:
        critical_depth = 0.0
    else:
        critical_depth = qc.depth(filter_function=lambda x: len(x[1]) > 1) / num_multiple_qubit_gates

    entanglement_ratio = num_multiple_qubit_gates / num_gates
    assert num_multiple_qubit_gates <= num_gates

    parallelism = (num_gates / depth - 1) / (qc.num_qubits - 1)

    liveness = liveness_A_matrix / (depth * qc.num_qubits)

    assert 0 <= program_communication <= 1
    assert 0 <= critical_depth <= 1
    assert 0 <= entanglement_ratio <= 1
    assert 0 <= parallelism <= 1
    assert 0 <= liveness <= 1

    return (
        program_communication,
        critical_depth,
        entanglement_ratio,
        parallelism,
        liveness,
    )


def crit_depth(qc: QuantumCircuit, precision: int = 10) -> float:
    (
        _,
        critical_depth,
        _,
        _,
        _,
    ) = calc_supermarq_features(qc)
    return cast(float, np.round(1 - critical_depth, precision))


def parallelism(qc: QuantumCircuit, precision: int = 10) -> float:
    (
        _,
        _,
        _,
        parallelism,
        _,
    ) = calc_supermarq_features(qc)
    return cast(float, np.round(1 - parallelism, precision))


def gate_ratio(qc: QuantumCircuit, precision: int = 10) -> float:
    return cast(float, np.round(1 - qc.num_nonlocal_gates() / qc.size(), precision))


def mix(qc: QuantumCircuit, device: Device, precision: int = 10) -> float:
    return expected_fidelity(qc, device, precision) * 0.5 + crit_depth(qc, precision) * 0.5


def expected_fidelity(qc_or_path: QuantumCircuit | Path, device: Device, precision: int = 10) -> float:
    if isinstance(qc_or_path, QuantumCircuit):
        qc = qc_or_path
    else:
        try:
            qc = QuantumCircuit.from_qasm_file(str(qc_or_path))
        except Exception:
            msg = "Could not read QuantumCircuit from: " + str(qc_or_path)
            raise RuntimeError(msg) from None

    res = 1.0

    for instruction, qargs, _cargs in qc.data:
        gate_type = instruction.name
        assert gate_type in device.basis_gates

        if gate_type == "barrier":
            continue

        if len(qargs) == 1:
            qubit = calc_qubit_index(qargs, qc.qregs, 0)
            if gate_type == "measure":
                res *= device.get_readout_fidelity(qubit)
            else:
                res *= device.get_single_qubit_gate_fidelity(gate_type, qubit)
        elif len(qargs) == 2:  # noqa: PLR2004
            qubit1 = calc_qubit_index(qargs, qc.qregs, 0)
            qubit2 = calc_qubit_index(qargs, qc.qregs, 1)
            res *= device.get_two_qubit_gate_fidelity(gate_type, qubit1, qubit2)
        else:
            msg = "Gate with more than 2 qubits is not supported: " + str(instruction)
            raise RuntimeError(msg)

    return cast(float, np.round(res, precision))
