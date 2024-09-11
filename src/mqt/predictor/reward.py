"""This module contains the functions to calculate the reward of a quantum circuit on a given device."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from mqt.bench.utils import calc_supermarq_features

if TYPE_CHECKING:
    from qiskit import QuantumCircuit, QuantumRegister, Qubit

    from mqt.bench.devices import Device

logger = logging.getLogger("mqt-predictor")

figure_of_merit = Literal["expected_fidelity", "critical_depth"]


def crit_depth(qc: QuantumCircuit, precision: int = 10) -> float:
    """Calculates the critical depth of a given quantum circuit."""
    supermarq_features = calc_supermarq_features(qc)
    return cast(float, np.round(1 - supermarq_features.critical_depth, precision))


def expected_fidelity(qc: QuantumCircuit, device: Device, precision: int = 10) -> float:
    """Calculates the expected fidelity of a given quantum circuit on a given device.

    Arguments:
        qc: The quantum circuit to be compiled.
        device: The device to be used for compilation.
        precision: The precision of the returned value. Defaults to 10.

    Returns:
        The expected fidelity of the given quantum circuit on the given device.
    """
    res = 1.0
    for qc_instruction in qc.data:
        instruction, qargs = qc_instruction.operation, qc_instruction.qubits
        gate_type = instruction.name

        if gate_type != "barrier":
            assert len(qargs) in [1, 2]
            first_qubit_idx = calc_qubit_index(qargs, qc.qregs, 0)

            if len(qargs) == 1:
                if gate_type == "measure":
                    specific_fidelity = device.get_readout_fidelity(first_qubit_idx)
                else:
                    specific_fidelity = device.get_single_qubit_gate_fidelity(gate_type, first_qubit_idx)
            else:
                second_qubit_idx = calc_qubit_index(qargs, qc.qregs, 1)
                specific_fidelity = device.get_two_qubit_gate_fidelity(gate_type, first_qubit_idx, second_qubit_idx)

            res *= specific_fidelity

    return cast(float, np.round(res, precision))


def calc_qubit_index(qargs: list[Qubit], qregs: list[QuantumRegister], index: int) -> int:
    """Calculates the global qubit index for a given quantum circuit and qubit index."""
    offset = 0
    for reg in qregs:
        if qargs[index] not in reg:
            offset += reg.size
        else:
            qubit_index: int = offset + reg.index(qargs[index])
            return qubit_index
    error_msg = f"Global qubit index for local qubit {index} index not found."
    raise ValueError(error_msg)
