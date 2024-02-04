from __future__ import annotations

import sys
from typing import Any

if sys.version_info < (3, 10, 0):
    import importlib_resources as resources
else:
    from importlib import resources  # type: ignore[no-redef]

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx  # type: ignore[import-untyped]
import numpy as np
import pyzx as zx  # type: ignore[import-not-found]
from joblib import dump
from qiskit import QuantumCircuit
from qiskit.circuit import Clbit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGInNode, DAGOutNode
from qiskit.transpiler.passes import RemoveBarriers
from torch_geometric.utils import from_networkx  # type: ignore[import-not-found]

from mqt.bench.devices import Device, get_available_devices
from mqt.bench.utils import calc_supermarq_features
from mqt.predictor import ml, reward, rl

if TYPE_CHECKING:
    import rustworkx as rx
    import torch_geometric  # type: ignore[import-not-found]
    from numpy._typing import NDArray
    from qiskit.circuit import Qubit
    from sklearn.ensemble import RandomForestClassifier


def qcompile(
    qc: QuantumCircuit,
    figure_of_merit: reward.figure_of_merit = "expected_fidelity",
    devices: list[Device] | None = None,
) -> tuple[QuantumCircuit, list[str], str]:
    """Compiles a given quantum circuit to a device with the highest predicted figure of merit.

    Args:
        qc (QuantumCircuit): The quantum circuit to be compiled.
        figure_of_merit (reward.reward_functions, optional): The figure of merit to be used for compilation. Defaults to "expected_fidelity".

    Returns:
        tuple[QuantumCircuit, list[str], str] | bool: Returns a tuple containing the compiled quantum circuit, the compilation information and the name of the device used for compilation. If compilation fails, False is returned.
    """

    device_name = predict_device_for_figure_of_merit(qc, figure_of_merit, devices)
    assert device_name is not None
    res = rl.qcompile(qc, figure_of_merit=figure_of_merit, device_name=device_name)
    return *res, device_name


def predict_device_for_figure_of_merit(
    qc: QuantumCircuit,
    figure_of_merit: reward.figure_of_merit = "expected_fidelity",
    devices: list[Device] | None = None,
) -> str:
    """Returns the name of the device with the highest predicted figure of merit that is suitable for the given quantum circuit.

    Args:
        qc (QuantumCircuit): The quantum circuit to be compiled.
        figure_of_merit (reward.reward_functions, optional): The figure of merit to be used for compilation. Defaults to "expected_fidelity".
        devices ([Device], optional): The devices to be considered for compilation.

    Returns:
        str : The name of the device with the highest predicted figure of merit that is suitable for the given quantum circuit.
    """
    if devices is None:
        devices = get_available_devices()

    ml_predictor = ml.Predictor(devices=devices)
    predicted_device_index_probs = ml_predictor.predict_probs(qc, figure_of_merit)
    assert ml_predictor.clf is not None
    classes = ml_predictor.clf.classes_  # type: ignore[unreachable]
    predicted_device_index = classes[np.argsort(predicted_device_index_probs)[::-1]]

    for index in predicted_device_index:
        if devices[index].num_qubits >= qc.num_qubits:
            return devices[index].name
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


def create_feature_dict(qc: str | QuantumCircuit) -> dict[str, Any]:
    """Creates and returns a feature dictionary for a given quantum circuit.

    Args:
        qc (str | QuantumCircuit): The quantum circuit to be compiled.

    Returns:
        dict[str, Any]: The feature dictionary of the given quantum circuit.
    """
    filename = qc
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

    # operations/gates encoding for graph feature creation
    ops_list_encoding = ops_list_dict.copy()
    ops_list_encoding["measure"] = len(ops_list_encoding)  # add extra gate
    # unique number for each gate {'measure': 0, 'cx': 1, ...}
    for i, key in enumerate(ops_list_dict):
        ops_list_encoding[key] = i

    feature_dict = {}
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
    feature_dict["graph"] = circuit_to_graph(qc, ops_list_encoding)
    feature_dict["zx_graph"] = qasm_to_zx(qc.qasm(), filename)
    return feature_dict


def save_classifier(clf: RandomForestClassifier, figure_of_merit: reward.figure_of_merit = "expected_fidelity") -> None:
    """Saves the given classifier to the trained model folder.

    Args:
        clf (RandomForestClassifier): The classifier to be saved.
        figure_of_merit (reward.reward_functions, optional): The figure of merit to be used for compilation. Defaults to "expected_fidelity".
    """
    dump(clf, str(get_path_trained_model(figure_of_merit)))


def save_training_data(
    training_data: list[NDArray[np.float_]],
    names_list: list[str],
    scores_list: list[NDArray[np.float_]],
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
) -> tuple[list[NDArray[np.float_]], list[str], list[NDArray[np.float_]]]:
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


def rustworkx_to_networkx(graph: rx.PyDAG[Any, Any], ops_list_encoding: dict[str, int]) -> nx.MultiDiGraph | nx.DiGraph:
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


def qasm_to_zx(qasm: str, filename: str) -> zx.Circuit:
    """
    Convert a qasm string to a zx-calculus string.
    """
    try:
        qasm = qasm.replace("pi", "3.141592653589793")
        qasm = qasm.replace("u(", "u1(")
        qasm = qasm.replace(
            'include "qelib1.inc";',
            'include "qelib1.inc";\n\n'
            "// controlled phase rotation\n"
            "gate cp(lambda) a,b\n"
            "{\n"
            "u1(lambda/2) a;\n"
            "cx a,b;\n"
            "u1(-lambda/2) b;\n"
            "cx a,b;\n"
            "u1(lambda/2) b;\n"
            "}",
        )

        return zx.Circuit.from_qasm(qasm)
    except Exception as e:
        print(filename)
        print(e)
        return zx.Circuit(0)


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
    X_train: NDArray[np.float_]
    X_test: NDArray[np.float_]
    y_train: NDArray[np.float_]
    y_test: NDArray[np.float_]
    indices_train: list[int]
    indices_test: list[int]
    names_list: list[str]
    scores_list: list[list[float]]
