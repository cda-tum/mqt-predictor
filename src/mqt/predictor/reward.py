"""This module contains the functions to calculate the reward of a quantum circuit on a given device."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from qiskit.transpiler import InstructionDurations, PassManager, passes

from mqt.bench.utils import calc_supermarq_features

if TYPE_CHECKING:
    from qiskit import QuantumCircuit, QuantumRegister, Qubit

    from mqt.bench.devices import Device

logger = logging.getLogger("mqt-predictor")

FIGURES_OF_MERIT = ["expected_fidelity", "critical_depth", "expected_success_probability"]
figure_of_merit = Literal["expected_fidelity", "critical_depth", "expected_success_probability"]


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
    for instruction, qargs, _cargs in qc.data:
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


def expected_success_probability(qc: QuantumCircuit, device: Device, precision: int = 10) -> float:
    """Calculates the expected success probability of a given quantum circuit on a given device.

    It is calculated by multiplying the expected fidelity with a min(T1,T2)-dependant
    decay factor during qubit idle times. Idle times are available after scheduling.

    Arguments:
        qc: The quantum circuit to be compiled.
        device: The device to be used for compilation.
        precision: The precision of the returned value. Defaults to 10.

    Returns:
        The expected success probability of the given quantum circuit on the given device.
    """

    # collect gate and measurement durations for active qubits
    op_times, active_qubits = [], set()
    try:
        for instruction, qargs, _cargs in qc.data:
            gate_type = instruction.name

            if gate_type == "barrier" or gate_type == "id":
                continue
            assert len(qargs) in [1, 2]
            first_qubit_idx = calc_qubit_index(qargs, qc.qregs, 0)
            active_qubits.add(first_qubit_idx)

            if len(qargs) == 1:  # single-qubit gate
                if gate_type == "measure":
                    duration = device.get_readout_duration(first_qubit_idx)
                else:
                    duration = device.get_single_qubit_gate_duration(gate_type, first_qubit_idx)
                op_times.append((gate_type, [first_qubit_idx], duration, "s"))
            else:  # multi-qubit gate
                second_qubit_idx = calc_qubit_index(qargs, qc.qregs, 1)
                active_qubits.add(second_qubit_idx)
                duration = device.get_two_qubit_gate_duration(gate_type, first_qubit_idx, second_qubit_idx)
                op_times.append((gate_type, [first_qubit_idx, second_qubit_idx], duration, "s"))
    except ValueError:
        logger.exception("Calculating ESP requires device with fully specified gate and readout durations.")
        return 0.0

    # associate gate and idle (delay) times for each qubit through asap scheduling
    sched_pass = passes.ASAPScheduleAnalysis(InstructionDurations(op_times))
    delay_pass = passes.PadDelay()
    pm = PassManager([sched_pass, delay_pass])
    scheduled_circ = pm.run(qc)

    res = 1.0
    for instruction, qargs, _cargs in scheduled_circ.data:
        gate_type = instruction.name

        if gate_type != "barrier" and gate_type != "id":
            assert len(qargs) in [1, 2]
            first_qubit_idx = calc_qubit_index(qargs, qc.qregs, 0)

            if len(qargs) == 1:
                if gate_type == "measure":
                    specific_fidelity = device.get_readout_fidelity(first_qubit_idx)
                elif gate_type == "delay":
                    # only consider active qubits
                    if first_qubit_idx in active_qubits:
                        idle_time = instruction.duration
                        T_min = min(
                            device.calibration.get_t1(first_qubit_idx), device.calibration.get_t2(first_qubit_idx)
                        )  # noqa:N806
                        specific_fidelity = np.exp(-idle_time / T_min)
                    else:
                        specific_fidelity = 1.0
                else:
                    specific_fidelity = device.get_single_qubit_gate_fidelity(gate_type, first_qubit_idx)
            else:
                second_qubit_idx = calc_qubit_index(qargs, qc.qregs, 1)
                specific_fidelity = device.get_two_qubit_gate_fidelity(gate_type, first_qubit_idx, second_qubit_idx)

            res *= specific_fidelity

    return cast(float, np.round(res, precision))
