from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx  # type: ignore[import-untyped]
import numpy as np
import pyzx as zx
from joblib import dump
from qiskit import QuantumCircuit
from qiskit.circuit import Clbit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGInNode, DAGOutNode
from qiskit.transpiler.passes import RemoveBarriers
from torch_geometric.utils import from_networkx

from mqt.bench.devices import Device, get_available_devices
from mqt.bench.utils import calc_supermarq_features
from mqt.predictor import ml, reward, rl

if TYPE_CHECKING:
    import rustworkx as rx
    import torch_geometric
    from numpy._typing import NDArray
    from qiskit.circuit import Qubit
    from sklearn.ensemble import RandomForestClassifier

    from mqt.bench.devices import Device


def qcompile(
    qc: QuantumCircuit,
    figure_of_merit: reward.figure_of_merit = "expected_fidelity",
) -> tuple[QuantumCircuit, list[str], str]:
    """Compiles a given quantum circuit to a device with the highest predicted figure of merit.

    Args:
        qc (QuantumCircuit): The quantum circuit to be compiled.
        figure_of_merit (reward.reward_functions, optional): The figure of merit to be used for compilation. Defaults to "expected_fidelity".

    Returns:
        tuple[QuantumCircuit, list[str], str] | bool: Returns a tuple containing the compiled quantum circuit, the compilation information and the name of the device used for compilation. If compilation fails, False is returned.
    """

    device = predict_device_for_figure_of_merit(qc, figure_of_merit)
    res = rl.qcompile(qc, figure_of_merit=figure_of_merit, device_name=device.name)
    return *res, device.name


def predict_device_for_figure_of_merit(
    qc: QuantumCircuit, figure_of_merit: reward.figure_of_merit = "expected_fidelity"
) -> Device:
    """Returns the name of the device with the highest predicted figure of merit that is suitable for the given quantum circuit.

    Args:
        qc (QuantumCircuit): The quantum circuit to be compiled.
        figure_of_merit (reward.reward_functions, optional): The figure of merit to be used for compilation. Defaults to "expected_fidelity".
        devices ([Device], optional): The devices to be considered for compilation.

    Returns:
        Device : The device with the highest predicted figure of merit that is suitable for the given quantum circuit.
    """
    ml_predictor = ml.Predictor()
    predicted_device_index_probs = ml_predictor.predict_probs(qc, figure_of_merit)
    assert ml_predictor.clf is not None
    classes = ml_predictor.clf.classes_  # type: ignore[unreachable]
    predicted_device_index = classes[np.argsort(predicted_device_index_probs)[::-1]]

    for index in predicted_device_index:
        if ml_predictor.devices[index].num_qubits >= qc.num_qubits:
            return ml_predictor.devices[index]
    msg = "No suitable device found."
    raise ValueError(msg)


def get_path_training_data() -> Path:
    """Returns the path to the training data folder."""
    return Path(str(resources.files("mqt.predictor"))) / "ml" / "training_data"


def get_path_results(ghz_results: bool = False) -> Path:
    """Returns the path to the results file."""
    if ghz_results:
        return get_path_training_data() / "trained_model" / "res_GHZ.csv"
    return get_path_training_data() / "trained_model" / "res.csv"


def get_path_trained_model(figure_of_merit: str, return_non_zero_indices: bool = False) -> Path:
    """Returns the path to the trained model folder resulting from the machine learning training."""
    if return_non_zero_indices:
        return get_path_training_data() / "trained_model" / ("non_zero_indices_" + figure_of_merit + ".npy")
    return get_path_training_data() / "trained_model" / ("trained_clf_" + figure_of_merit + ".joblib")


def get_path_training_circuits() -> Path:
    """Returns the path to the training circuits folder."""
    return get_path_training_data() / "training_circuits"


def get_path_training_circuits_compiled() -> Path:
    """Returns the path to the compiled training circuits folder."""
    return get_path_training_data() / "training_circuits_compiled"


def get_openqasm_gates() -> list[str]:
    """Returns a list of all quantum gates within the openQASM 2.0 standard header."""
    # according to https://github.com/Qiskit/qiskit-terra/blob/main/qiskit/qasm/libs/qelib1.inc
    return [
        "u3",
        "u2",
        "u1",
        "cx",
        "id",
        "u0",
        "u",
        "p",
        "x",
        "y",
        "z",
        "h",
        "s",
        "sdg",
        "t",
        "tdg",
        "rx",
        "ry",
        "rz",
        "sx",
        "sxdg",
        "cz",
        "cy",
        "swap",
        "ch",
        "ccx",
        "cswap",
        "crx",
        "cry",
        "crz",
        "cu1",
        "cp",
        "cu3",
        "csx",
        "cu",
        "rxx",
        "rzz",
        "rccx",
        "rc3x",
        "c3x",
        "c3sqrtx",
        "c4x",
    ]


def dict_to_featurevector(gate_dict: dict[str, int]) -> dict[str, int]:
    """Calculates and returns the feature vector of a given quantum circuit gate dictionary."""
    res_dct = dict.fromkeys(get_openqasm_gates(), 0)
    for key, val in dict(gate_dict).items():
        if key in res_dct:
            res_dct[key] = val

    return res_dct


PATH_LENGTH = 260


def create_feature_dict(qc: str | QuantumCircuit, graph_features: bool = False) -> dict[str, Any]:
    """Creates and returns a feature dictionary for a given quantum circuit.

    Args:
        qc (str | QuantumCircuit): The quantum circuit to be compiled.
        graph_features (bool): Whether to include graph features in the feature dictionary.

    Returns:
        dict[str, Any]: The feature dictionary of the given quantum circuit.
    """
    if not isinstance(qc, QuantumCircuit):
        if len(qc) < PATH_LENGTH and Path(qc).exists():
            qc = QuantumCircuit.from_qasm_file(qc)
        elif "OPENQASM" in qc:
            qc = QuantumCircuit.from_qasm_str(qc)
        else:
            error_msg = "Invalid input for 'qc' parameter."
            raise ValueError(error_msg) from None

    ops_list = qc.count_ops()
    ops_list_dict = dict_to_featurevector(ops_list)
    feature_dict = {}

    if graph_features:
        try:
            # operations/gates encoding for graph feature creation
            ops_list_encoding = ops_list_dict.copy()
            ops_list_encoding["measure"] = len(ops_list_encoding)  # add extra gate
            # unique number for each gate {'measure': 0, 'cx': 1, ...}
            for i, key in enumerate(ops_list_dict):
                ops_list_encoding[key] = i
            feature_dict["graph"] = circuit_to_graph(qc, ops_list_encoding)
        except Exception:
            feature_dict["graph"] = None
        try:
            feature_dict["zx_graph"] = qasm_to_zx(qc.qasm())
        except Exception:  # e.g. zx-calculus not supported for all circuits
            feature_dict["zx_graph"] = None

    for key in ops_list_dict:
        feature_dict[key] = float(ops_list_dict[key])

    feature_dict["num_qubits"] = float(qc.num_qubits)
    feature_dict["depth"] = float(qc.depth())

    supermarq_features = calc_supermarq_features(qc)
    feature_dict["program_communication"] = supermarq_features.program_communication
    feature_dict["critical_depth"] = supermarq_features.critical_depth
    feature_dict["entanglement_ratio"] = supermarq_features.entanglement_ratio
    feature_dict["parallelism"] = supermarq_features.parallelism
    feature_dict["liveness"] = supermarq_features.liveness
    feature_dict["directed_program_communication"] = supermarq_features.directed_program_communication
    feature_dict["gate_coverage"] = supermarq_features.directed_critical_depth
    feature_dict["singleQ_gates_per_layer"] = supermarq_features.singleQ_gates_per_layer
    feature_dict["multiQ_gates_per_layer"] = supermarq_features.multiQ_gates_per_layer
    feature_dict["my_critical_depth"] = supermarq_features.my_critical_depth
    return feature_dict


def save_classifier(clf: RandomForestClassifier, figure_of_merit: reward.figure_of_merit = "expected_fidelity") -> None:
    """Saves the given classifier to the trained model folder.

    Args:
        clf (RandomForestClassifier): The classifier to be saved.
        figure_of_merit (reward.reward_functions, optional): The figure of merit to be used for compilation. Defaults to "expected_fidelity".
    """
    dump(clf, str(get_path_trained_model(figure_of_merit)))


def save_training_data(
    training_data: list[NDArray[np.float64]],
    names_list: list[str],
    scores_list: list[NDArray[np.float64]],
    figure_of_merit: reward.figure_of_merit,
) -> None:
    """Saves the given training data to the training data folder.

    Args:
        res (tuple[list[Any], list[Any], list[Any]]): The training data, the names list and the scores list to be saved.
        figure_of_merit (reward.reward_functions, optional): The figure of merit to be used for compilation. Defaults to "expected_fidelity".
    """

    with resources.as_file(get_path_training_data() / "training_data_graph") as path:
        data = np.asarray(training_data, dtype=object)
        np.save(str(path / ("training_data_" + figure_of_merit + ".npy")), data)
        data = np.asarray(names_list)
        np.save(str(path / ("names_list_" + figure_of_merit + ".npy")), data)
        data = np.asarray(scores_list)
        np.save(str(path / ("scores_list_" + figure_of_merit + ".npy")), data)


def load_training_data(
    figure_of_merit: reward.figure_of_merit = "expected_fidelity",
) -> tuple[list[NDArray[np.float64]], list[str], list[NDArray[np.float64]]]:
    """Loads and returns the training data from the training data folder.

    Args:
        figure_of_merit (reward.reward_functions, optional): The figure of merit to be used for compilation. Defaults to "expected_fidelity".

    Returns:
       tuple[NDArray[np.float_], list[str], list[NDArray[np.float_]]]: The training data, the names list and the scores list.
    """
    with resources.as_file(get_path_training_data() / "training_data_graph") as path:
        if (
            path.joinpath("training_data_" + figure_of_merit + ".npy").is_file()
            and path.joinpath("names_list_" + figure_of_merit + ".npy").is_file()
            and path.joinpath("scores_list_" + figure_of_merit + ".npy").is_file()
        ):
            training_data = np.load(path / ("training_data_" + figure_of_merit + ".npy"), allow_pickle=True)
            names_list = list(np.load(path / ("names_list_" + figure_of_merit + ".npy"), allow_pickle=True))
            scores_list = list(np.load(path / ("scores_list_" + figure_of_merit + ".npy"), allow_pickle=True))
        else:
            error_msg = "Training data not found. Please run the training script first."
            raise FileNotFoundError(error_msg)

        return training_data, names_list, scores_list


def rustworkx_to_networkx(graph: rx.PyDAG[Any, Any], ops_list_encoding: dict[str, int]) -> nx.DiGraph:
    """
    Convert a rustworkx DAG to a networkx graph.
    """
    # create a networkx graph
    nx_graph = nx.DiGraph()

    control_dict = {}  # to mark edges as control
    # add node operations = gates as nodes
    for node_idx, node in enumerate(graph.nodes()):
        if type(node) in [DAGInNode, DAGOutNode]:
            op = ops_list_encoding["id"]
        else:
            op = ops_list_encoding[node.op.name]

            controls = []  # mark controls
            if hasattr(node.op, "num_ctrl_qubits"):
                # last qubit is target
                controls += list(node.qargs[:-1])
            if node.op.condition:
                # classical c_if(...) operation
                controls += node.op.condition_bits
            if controls:
                control_dict[node_idx] = controls

        nx_graph.add_node(node_idx, gate=op)

    count = 0
    bit_dict: dict[Qubit | Clbit, int] = {}
    # add quantum/classical bits as edges
    for edge in graph.weighted_edge_list():
        source, target, bit = edge

        if not bit_dict.get(bit, False):
            bit_dict[bit] = count
            count += 1
        # bit_nr = bit_dict[bit]

        is_classic = 1 if isinstance(bit, Clbit) else 0
        is_control = 1 if bit in control_dict.get(target, []) else 0

        if is_classic == 0:  # only add quantum wires
            nx_graph.add_edge(source, target, is_control=is_control)  # , bit_nr=bit_nr, is_classic=is_classic)

    return nx_graph


def zx_to_networkx(zx_graph: zx.graph.BaseGraph) -> nx.DiGraph:
    # create a networkx graph
    nx_graph = nx.DiGraph()

    # Add nodes to the NetworkX graph
    for idx in zx_graph.vertices():
        t = zx_graph.type(idx)
        phase = float(zx_graph.phase(idx))

        nx_graph.add_node(idx, gate=t, phase=phase)

    # Add edges to the NetworkX graph
    for e in zx_graph.edges():
        source, target = e
        t = zx_graph.edge_type(e)

        nx_graph.add_edge(source, target, wire=t)

    return nx_graph


def substitute_cp_and_cu1_gate(qasm: str) -> str:
    """
    Substitute all occurrences of the cp gate in a qasm string with a custom gate definition.
    """

    # Function to replace cp gate with custom gate definition
    def replace_cp_gate(match: re.Match[str]) -> str:
        phase, qubit1, qubit2 = match.groups()
        if "/" in phase:
            numerator, denominator = phase.split("/")
            new_denominator = str(int(denominator) * 2)
            phase_2 = numerator + "/" + new_denominator
        else:
            phase_2 = phase + "/2"

        # rz is same as u1 up to a global phase
        return (
            f"rz({phase_2}) {qubit2};\n"
            f"cx {qubit1},{qubit2};\n"
            f"rz(-{phase_2}) {qubit2};\n"
            f"cx {qubit1},{qubit2};\n"
            f"rz({phase_2}) {qubit2};\n"
        )

    # Replace all occurrences of the cp gate with the custom gate definition
    # cp is same as cu1
    qasm = re.sub(r"cp\((.+?)\) (.+?),(.+?);", replace_cp_gate, qasm)
    # Replace all occurrences of the cp gate with the custom gate definition
    qasm = re.sub(r"cu1\((.+?)\) (.+?),(.+?);", replace_cp_gate, qasm)

    return qasm.replace("--", "")


def format_u1_gate(qasm: str) -> str:
    def format_u1(match: re.Match[str]) -> str:
        phase = match.group(1)
        return f"u1({phase})"

    return re.sub(r"u\(0,0,(.+?)\)", format_u1, qasm)


def replace_swap_gate(qasm: str) -> str:
    def format_swap(match: re.Match[str]) -> str:
        qubit1 = match.group(1)
        qubit2 = match.group(2)
        return f"cx {qubit1},{qubit2}; cx {qubit2},{qubit1}; cx {qubit1},{qubit2};"

    return re.sub(r"swap (.+?),(.+?);", format_swap, qasm)


def qasm_to_zx(qasm: str) -> zx.Circuit:
    """
    Convert a qasm string to a zx-calculus string.
    """
    qasm = substitute_cp_and_cu1_gate(qasm)
    qasm = format_u1_gate(qasm)
    qasm = replace_swap_gate(qasm)
    qasm = qasm.replace("u1(", "rz(")
    qasm = qasm.replace("p(", "rz(")

    try:
        zx_circ = zx.Circuit.from_qasm(qasm)
        zx_graph = zx_circ.to_graph()

        nx_graph = zx_to_networkx(zx_graph)

        #### Postprocessing ###
        # Remove root and leaf nodes (in and out nodes)
        nodes_to_remove = [node for node, degree in nx_graph.degree() if degree < 1]
        nx_graph.remove_nodes_from(nodes_to_remove)

        # Convert to torch_geometric data
        return from_networkx(nx_graph, group_node_attrs=all, group_edge_attrs=all)
    except Exception as e:
        msg = f"Error in qasm_to_zx: {e}"
        raise ValueError(msg) from e


def circuit_to_graph(qc: QuantumCircuit, ops_list_encoding: dict[str, int]) -> torch_geometric.data.Data:
    """
    Convert a quantum circuit to a torch_geometric graph.
    """
    ### Preprocessing ###
    circ = RemoveBarriers()(qc)

    # Convert to a rustworkx DAG
    dag_circuit = circuit_to_dag(circ, copy_operations=False)
    dag_graph = dag_circuit._multi_graph  # noqa: SLF001
    # Convert to networkx graph
    nx_graph = rustworkx_to_networkx(dag_graph, ops_list_encoding)

    #### Postprocessing ###
    # Remove root and leaf nodes (in and out nodes)
    nodes_to_remove = [node for node, degree in nx_graph.degree() if degree <= 1]
    nx_graph.remove_nodes_from(nodes_to_remove)

    # Convert to torch_geometric data
    return from_networkx(nx_graph, group_node_attrs=all, group_edge_attrs=all)


@dataclass
class TrainingData:
    X_train: NDArray[np.float64]
    X_test: NDArray[np.float64]
    y_train: NDArray[np.float64]
    y_test: NDArray[np.float64]
    indices_train: list[int]
    indices_test: list[int]
    names_list: list[str]
    scores_list: list[list[float]]
