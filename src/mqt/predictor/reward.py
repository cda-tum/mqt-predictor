from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from qiskit.compiler import transpile

from mqt.bench.utils import calc_supermarq_features

if TYPE_CHECKING:
    from qiskit import QuantumCircuit, QuantumRegister, Qubit

    from mqt.bench.devices import Device

logger = logging.getLogger("mqt-predictor")

figure_of_merit = Literal["expected_fidelity", "critical_depth", "expected_success_probability"]


def crit_depth(qc: QuantumCircuit, precision: int = 10) -> float:
    """Calculates the critical depth of a given quantum circuit."""
    supermarq_features = calc_supermarq_features(qc)
    return cast(float, np.round(1 - supermarq_features.critical_depth, precision))


def expected_fidelity(qc: QuantumCircuit, device: Device, precision: int = 10) -> float:
    """Calculates the expected fidelity of a given quantum circuit on a given device.

    Args:
        qc (QuantumCircuit): The quantum circuit to be compiled.
        device(mqt.bench.Device): The device to be used for compilation.
        precision (int, optional): The precision of the returned value. Defaults to 10.

    Returns:
        float: The expected fidelity of the given quantum circuit on the given device.
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

    Args:
        qc (QuantumCircuit): The quantum circuit to be compiled.
        device(mqt.bench.Device): The device to be used for compilation.
        precision (int, optional): The precision of the returned value. Defaults to 10.

    Returns:
        float: The expected success probability of the given quantum circuit on the given device.
    """
    res = 1.0

    # collect gate durations for all qubits
    op_times = []
    for op in device.basis_gates:
        if op == "barrier" or op == "id":
            continue
        if op in ["rxx", "rzz", "cx", "cz", "cp", "ecr", "xx_plus_yy"]:
            for q0, q1 in device.coupling_map:  # multi-qubit gates
                try:
                    d = device.get_two_qubit_gate_duration(op, q0, q1)
                    op_times.append((op, [q0, q1], d, "s"))
                except ValueError:  # noqa:PERF203
                    pass  # gate time not available for this qubit pair
        else:  # single-qubit gates
            for q in range(device.num_qubits):
                try:
                    if op == "measure":
                        d = device.get_readout_duration(q)
                    else:
                        d = device.get_single_qubit_gate_duration(op, q)
                    op_times.append((op, [q], d, "s"))
                except ValueError:  # noqa:PERF203
                    pass  # gate time not available for this qubit

    # associate gate times for each qubit through asap scheduling
    transpiled = transpile(qc, scheduling_method="asap", basis_gates=device.basis_gates, instruction_durations=op_times)

    qubit_durations: dict[int, list[float]] = {qubit_index: [] for qubit_index in range(device.num_qubits)}

    for instruction, qargs, _cargs in transpiled.data:
        gate_type = instruction.name
        if gate_type == "barrier":
            continue

        assert len(qargs) in [1, 2]
        first_qubit_idx = calc_qubit_index(qargs, qc.qregs, 0)
        if device.name == "rigetti_aspen_m2":
            first_qubit_idx = (first_qubit_idx + 40) % 80

        fidelity = 1.0
        if len(qargs) == 1:  # single-qubit gate
            if gate_type == "measure":  # measurement reliability
                fidelity = device.get_readout_fidelity(first_qubit_idx)
            elif gate_type == "delay":
                pass  # idle: only T1,T2 decoherence
            else:  # gate reliability
                fidelity = device.get_single_qubit_gate_fidelity(gate_type, first_qubit_idx)
            qubit_durations[first_qubit_idx].append(instruction.duration)

        else:  # multi-qubit gate
            second_qubit_idx = calc_qubit_index(qargs, qc.qregs, 1)
            if device.name == "rigetti_aspen_m2":
                second_qubit_idx = (second_qubit_idx + 40) % 80

            fidelity = device.get_two_qubit_gate_fidelity(gate_type, first_qubit_idx, second_qubit_idx)
            qubit_durations[first_qubit_idx].append(instruction.duration)
            qubit_durations[second_qubit_idx].append(instruction.duration)

        res *= fidelity

    # calculate T1,T2 decoherence over entire qubit lifetime
    for qubit_idx, durations in qubit_durations.items():
        # don't consider idle qubits
        if len(durations) <= 1:
            continue
        lifetime = sum(durations)  # all operation times on qubit
        # only use dominant decoherence term (either T1 or T2)
        T_min = min(device.calibration.get_t1(qubit_idx), device.calibration.get_t2(qubit_idx))
        decoherence = np.exp(-lifetime / T_min)
        res *= decoherence

    return cast(float, np.round(res, precision))
