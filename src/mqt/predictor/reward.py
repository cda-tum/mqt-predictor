"""This module contains the functions to calculate the reward of a quantum circuit on a given device."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from mqt.bench.utils import calc_supermarq_features

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from qiskit import QuantumRegister, Qubit

    from mqt.bench.devices import Device

logger = logging.getLogger("mqt-predictor")

figure_of_merit = Literal[
    "expected_fidelity",
    "critical_depth",
    "estimated_success_probability",
    "estimated_hellinger_distance",
]


def crit_depth(qc: QuantumCircuit, precision: int = 10) -> float:
    """Calculates the critical depth of a given quantum circuit."""
    supermarq_features = calc_supermarq_features(qc)
    return cast("float", np.round(1 - supermarq_features.critical_depth, precision))


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

    return cast("float", np.round(res, precision))


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


def estimated_success_probability(qc: QuantumCircuit, device: Device, precision: int = 10) -> float:
    """Calculates the estimated success probability of a given quantum circuit on a given device.

    It is calculated by multiplying the expected fidelity with a min(T1,T2)-dependent
    decay factor during qubit idle times. To this end, the circuit is scheduled using ASAP scheduling.

    Arguments:
        qc: The quantum circuit to be compiled.
        device: The device to be used for compilation.
        precision: The precision of the returned value. Defaults to 10.

    Returns:
        The expected success probability of the given quantum circuit on the given device.
    """
    # lazy import of qiskit transpiler
    from qiskit.transpiler import InstructionDurations, Layout, PassManager, passes  # noqa: PLC0415
    from qiskit.transpiler.passes import ApplyLayout, SetLayout  # noqa: PLC0415

    # collect gate and measurement durations for active qubits
    op_times, active_qubits = [], set()
    for instruction, qargs, _cargs in qc.data:
        gate_type = instruction.name

        if gate_type == "barrier" or gate_type == "id":
            continue
        assert len(qargs) in (1, 2)
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

    # check whether the circuit was transformed by tket (i.e. changed register name)
    # qiskit ASAPScheduleAnalysis expects all qubit registers to be named 'q'
    if qc.qregs[0].name != "q":
        # create a layout that maps the (tket) 'node' registers to the (qiskit) 'q' registers
        layouts = [SetLayout(Layout({node_qubit: i for i, node_qubit in enumerate(node_reg)})) for node_reg in qc.qregs]
        # create a pass manager with the SetLayout and ApplyLayout passes
        pm = PassManager(list(layouts))
        pm.append(ApplyLayout())

        # replace the 'node' register with the 'q' register in the circuit
        qc = pm.run(qc)
        assert qc.qregs[0].name == "q"

    # associate gate and idle (delay) times for each qubit through asap scheduling
    sched_pass = passes.ASAPScheduleAnalysis(InstructionDurations(op_times))
    delay_pass = passes.PadDelay()
    pm = PassManager([sched_pass, delay_pass])
    scheduled_circ = pm.run(qc)

    res = 1.0
    for instruction, qargs, _cargs in scheduled_circ.data:
        gate_type = instruction.name

        if gate_type == "barrier" or gate_type == "id":
            continue

        assert len(qargs) in (1, 2)
        first_qubit_idx = calc_qubit_index(qargs, qc.qregs, 0)

        if len(qargs) == 1:
            if gate_type == "measure":
                res *= device.get_readout_fidelity(first_qubit_idx)
                continue

            if gate_type == "delay":
                # only consider active qubits
                if first_qubit_idx not in active_qubits:
                    continue

                res *= np.exp(
                    -instruction.duration
                    / min(device.calibration.get_t1(first_qubit_idx), device.calibration.get_t2(first_qubit_idx))
                )
                continue

            res *= device.get_single_qubit_gate_fidelity(gate_type, first_qubit_idx)
        else:
            second_qubit_idx = calc_qubit_index(qargs, qc.qregs, 1)
            res *= device.get_two_qubit_gate_fidelity(gate_type, first_qubit_idx, second_qubit_idx)

    return cast("float", np.round(res, precision))


def esp_data_available(device: Device) -> bool:
    """Check if calibration data to calculate ESP is available for the device."""

    def message(calibration: str, operation: str, target: int | str) -> str:
        return f"{calibration} data for {operation} operation on qubit(s) {target} is required to calculate ESP for device {device.name}."

    for qubit in range(device.num_qubits):
        try:
            device.calibration.get_t1(qubit)
        except ValueError:
            logger.exception(message("T1", "idle", qubit))
            return False
        try:
            device.calibration.get_t2(qubit)
        except ValueError:
            logger.exception(message("T2", "idle", qubit))
            return False
        try:
            device.get_readout_fidelity(qubit)
        except ValueError:
            logger.exception(message("Fidelity", "readout", qubit))
            return False
        try:
            device.get_readout_duration(qubit)
        except ValueError:
            logger.exception(message("Duration", "readout", qubit))
            return False

        for gate in device.get_single_qubit_gates():
            try:
                device.get_single_qubit_gate_fidelity(gate, qubit)
            except ValueError:
                logger.exception(message("Fidelity", gate, qubit))
                return False
            try:
                device.get_single_qubit_gate_duration(gate, qubit)
            except ValueError:
                logger.exception(message("Duration", gate, qubit))
                return False

    for gate in device.get_two_qubit_gates():
        for edge in device.coupling_map:
            try:
                device.get_two_qubit_gate_fidelity(gate, edge[0], edge[1])
            except ValueError:
                logger.exception(message("Fidelity", gate, edge))
                return False
            try:
                device.get_two_qubit_gate_duration(gate, edge[0], edge[1])
            except ValueError:
                logger.exception(message("Duration", gate, edge))
                return False

    return True


def calc_device_specific_features(
    qc: QuantumCircuit, device: Device, ignore_gates: list[str] | None = None
) -> dict[str, float]:
    """Creates and returns a qpu specific feature dictionary for a given quantum circuit and device.

    Arguments:
        qc: The quantum circuit for which the features are calculated.
        device: The device for which the features are calculated.
        ignore_gates: A list of gates to be ignored when calculating the features. Defaults to ["barrier", "id"].

    Returns:
        The device-specific feature dictionary of the given quantum circuit consisting of:
        - The number of operations for each native gate (excluding `ignore_gates`)
        - The active qubits (one-hot encoded e.g. {Qubit1: 1, Qubit2: 0, Qubit3: 0})
        - The depth of the quantum circuit
        - The number of qubits active in the computations
        - The supermarq features (wo program communication)
        - The directed program communication
        - The single and multi qubit gate ratio
    """
    if ignore_gates is None:
        ignore_gates = ["barrier", "id", "measure"]

    # Prepare circuit
    if qc.num_qubits != device.num_qubits:
        # Must have same number of qubits as the device
        circ = QuantumCircuit(device.num_qubits)
        circ.compose(qc, inplace=True)
    else:
        circ = qc.copy()

    # Create a dictionary with all native gates
    native_gate_dict = {gate: 0.0 for gate in device.basis_gates if gate not in ignore_gates}
    # Add the number of operations for each native gate
    for key, val in circ.count_ops().items():
        if key in native_gate_dict:
            native_gate_dict[key] = val
    feature_dict = native_gate_dict

    # Create a list of zeros for the one-hot vector
    active_qubits_dict = dict.fromkeys(circ.qubits, 0)
    # Iterate over the operations in the quantum circuit
    for op in circ.data:
        if op.operation.name in ignore_gates:
            continue
        # Mark the qubits that are used in the operation as active
        for qubit in op.qubits:
            active_qubits_dict[qubit] = 1
    feature_dict.update(active_qubits_dict)

    # Add the depth of the quantum circuit to the feature dictionary
    feature_dict["depth"] = circ.depth()

    # Add the number of active qubits to the feature dictionary
    num_active_qubits = sum(active_qubits_dict.values())
    feature_dict["num_qubits"] = num_active_qubits  # NOTE: not used in feature vector

    # Calculate supermarq features, which uses circ.num_qubits
    assert device.num_qubits == circ.num_qubits
    supermarq_features = calc_supermarq_features(circ)
    feature_dict["critical_depth"] = supermarq_features.critical_depth
    feature_dict["entanglement_ratio"] = supermarq_features.entanglement_ratio
    feature_dict["parallelism"] = supermarq_features.parallelism
    feature_dict["liveness"] = supermarq_features.liveness

    # NOTE: renormalize using the number of active qubits (not the device's num_qubits)
    feature_dict["parallelism"] = (
        feature_dict["parallelism"] * (device.num_qubits - 1) / (num_active_qubits - 1) if num_active_qubits >= 2 else 0
    )
    feature_dict["liveness"] = (
        feature_dict["liveness"] * device.num_qubits / num_active_qubits if num_active_qubits >= 1 else 0
    )

    # Calculate additional features based on DAG
    dag = circuit_to_dag(circ)
    dag.remove_all_ops_named("barrier")
    dag.remove_all_ops_named("measure")

    # Directed program communication = circuit's average directed qubit degree / degree of a complete directed graph.
    di_graph = nx.DiGraph()
    for op in dag.two_qubit_ops():
        q1, q2 = op.qargs
        di_graph.add_edge(circ.find_bit(q1).index, circ.find_bit(q2).index)
    degree_sum = sum(di_graph.degree(n) for n in di_graph.nodes)
    directed_program_communication = (
        degree_sum / (2 * num_active_qubits * (num_active_qubits - 1)) if num_active_qubits >= 2 else 0
    )

    # Average number of 1q gates per layer = num of 1-qubit gates in the circuit / depth
    single_qubit_gates_per_layer = (
        (len(dag.gate_nodes()) - len(dag.two_qubit_ops())) / dag.depth() if dag.depth() > 0 else 0
    )
    # Average number of 2q gates per layer = num of 2-qubit gates in the circuit / depth
    multi_qubit_gates_per_layer = len(dag.two_qubit_ops()) / dag.depth() if dag.depth() > 0 else 0

    # Normalize both values using the available number of qubits
    single_qubit_gates_per_layer = single_qubit_gates_per_layer / num_active_qubits if num_active_qubits > 1 else 0
    multi_qubit_gates_per_layer = multi_qubit_gates_per_layer / (num_active_qubits // 2) if num_active_qubits > 2 else 0

    feature_dict["directed_program_communication"] = directed_program_communication
    feature_dict["single_qubit_gates_per_layer"] = single_qubit_gates_per_layer
    feature_dict["multi_qubit_gates_per_layer"] = multi_qubit_gates_per_layer

    return feature_dict


def hellinger_distance(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    """Calculates the Hellinger distance between two probability distributions."""
    assert np.isclose(np.sum(p), 1, 0.05), "p is not a probability distribution"
    assert np.isclose(np.sum(q), 1, 0.05), "q is not a probability distribution"

    return float((1 / np.sqrt(2)) * np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))


def estimated_hellinger_distance(qc: QuantumCircuit, device: Device, precision: int = 10) -> float:
    """Calculates the estimated Hellinger distance of a given quantum circuit on a given device.

    Arguments:
        qc: The quantum circuit to be compiled.
        device: The device to be used for compilation.
        precision: The precision of the returned value. Defaults to 10.

    Returns:
        The estimated Hellinger distance of the given quantum circuit on the given device.
    """
    path = Path.cwd() / "evaluations" / "zenodo" / "trained_models"
    non_zero_indices_path = path / f"{device.name}_non_zero_indices.pkl"
    model_path = path / f"{device.name}_rf_regressor.pkl"

    # load model and vector of non-zero indices from files
    with Path.open(non_zero_indices_path, "rb") as f:
        non_zero_indices = pickle.load(f)
    with Path.open(model_path, "rb") as f:
        model = pickle.load(f)

    feature_dict = calc_device_specific_features(qc, device)
    feature_vector = list(feature_dict.values())
    # adjust the feature vector according to the non-zero indices
    feature_vector = [feature_vector[i] for i in non_zero_indices.values() if i]

    res = model.predict([feature_vector])[0]
    return cast("float", np.round(res, precision))


def hellinger_model_available(device: Device) -> bool:
    """Check if a trained model to estimate the Hellinger distance is available for the device."""
    path = Path.cwd() / "evaluations" / "zenodo" / "trained_models"
    path = path / f"{device.name}_rf_regressor.pkl"
    return bool(path.is_file())
