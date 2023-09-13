from __future__ import annotations

import sys
from typing import Any

if sys.version_info < (3, 10, 0):
    import importlib_resources as resources
else:
    from importlib import resources  # type: ignore[no-redef]

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from joblib import dump
from mqt.bench.utils import calc_supermarq_features
from mqt.predictor import ml, reward, rl
from qiskit import QuantumCircuit

if TYPE_CHECKING:
    from sklearn.ensemble import RandomForestClassifier


def qcompile(
    qc: QuantumCircuit, figure_of_merit: reward.reward_functions = "fidelity"
) -> tuple[QuantumCircuit, list[str], str] | bool:
    """Compiles a given quantum circuit to a device with the highest predicted figure of merit.

    Args:
        qc (QuantumCircuit): The quantum circuit to be compiled.
        figure_of_merit (reward.reward_functions, optional): The figure of merit to be used for compilation. Defaults to "fidelity".

    Returns:
        tuple[QuantumCircuit, list[str], str] | bool: Returns a tuple containing the compiled quantum circuit, the compilation information and the name of the device used for compilation. If compilation fails, False is returned.
    """

    device_name = get_predicted_and_suitable_device_name(qc, figure_of_merit)
    assert device_name is not None
    res = rl.qcompile(qc, figure_of_merit=figure_of_merit, device_name=device_name)
    if res:
        assert isinstance(res, tuple)
        return *res, device_name
    return False


def get_predicted_and_suitable_device_name(
    qc: QuantumCircuit, figure_of_merit: reward.reward_functions = "fidelity"
) -> str | None:
    """Returns the name of the device with the highest predicted figure of merit that is suitable for the given quantum circuit.

    Args:
        qc (QuantumCircuit): The quantum circuit to be compiled.
        figure_of_merit (reward.reward_functions, optional): The figure of merit to be used for compilation. Defaults to "fidelity".

    Returns:
        str | None: The name of the device with the highest predicted figure of merit that is suitable for the given quantum circuit. If no device is suitable, None is returned.
    """

    ml_predictor = ml.Predictor()
    predicted_device_index_probs = ml_predictor.predict_probs(qc, figure_of_merit)
    assert ml_predictor.clf is not None
    classes = ml_predictor.clf.classes_  # type: ignore[unreachable]
    predicted_device_index = classes[np.argsort(predicted_device_index_probs)[::-1]]
    devices = rl.helper.get_devices()

    for index in predicted_device_index:
        if devices[index]["max_qubits"] >= qc.num_qubits:
            return devices[index]["name"]
    return None


def get_path_training_data() -> Path:
    """Returns the path to the training data folder."""
    return Path(str(resources.files("mqt.predictor"))) / "ml" / "training_data"


def get_path_trained_model() -> Path:
    """Returns the path to the trained model folder resulting from the machine learning training."""
    return get_path_training_data() / "trained_model"


def get_path_training_circuits() -> Path:
    """Returns the path to the training circuits folder."""
    return get_path_training_data() / "training_circuits"


def get_path_training_circuits_compiled() -> Path:
    """Returns the path to the compiled training circuits folder."""
    return get_path_training_data() / "training_circuits_compiled"


def get_index_to_device_LUT() -> dict[int, str]:
    """Returns a look-up table (LUT) that maps the index of a device to its name."""
    devices = rl.helper.get_devices()
    return {i: device["name"] for i, device in enumerate(devices)}


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
    return feature_dict


def save_classifier(clf: RandomForestClassifier, figure_of_merit: reward.reward_functions = "fidelity") -> None:
    """Saves the given classifier to the trained model folder.

    Args:
        clf (RandomForestClassifier): The classifier to be saved.
        figure_of_merit (reward.reward_functions, optional): The figure of merit to be used for compilation. Defaults to "fidelity".
    """
    dump(clf, str(get_path_trained_model() / ("trained_clf_" + figure_of_merit + ".joblib")))


def save_training_data(
    res: tuple[list[Any], list[Any], list[Any]], figure_of_merit: reward.reward_functions = "fidelity"
) -> None:
    """Saves the given training data to the training data folder.

    Args:
        res (tuple[list[Any], list[Any], list[Any]]): The training data, the names list and the scores list to be saved.
        figure_of_merit (reward.reward_functions, optional): The figure of merit to be used for compilation. Defaults to "fidelity".
    """

    training_data, names_list, scores_list = res

    with resources.as_file(get_path_training_data() / "training_data_aggregated") as path:
        data = np.asarray(training_data, dtype=object)
        np.save(str(path / ("training_data_" + figure_of_merit + ".npy")), data)
        data = np.asarray(names_list)
        np.save(str(path / ("names_list_" + figure_of_merit + ".npy")), data)
        data = np.asarray(scores_list)
        np.save(str(path / ("scores_list_" + figure_of_merit + ".npy")), data)


def load_training_data(
    figure_of_merit: reward.reward_functions = "fidelity",
) -> tuple[list[Any], list[str], list[Any]]:
    """Loads and returns the training data from the training data folder.

    Args:
        figure_of_merit (reward.reward_functions, optional): The figure of merit to be used for compilation. Defaults to "fidelity".

    Returns:
        tuple[list[Any], list[str], list[Any]]: The training data, the names list and the scores list.
    """
    with resources.as_file(get_path_training_data() / "training_data_aggregated") as path:
        if (
            path.joinpath("training_data_" + figure_of_merit + ".npy").is_file()
            and path.joinpath("names_list_" + figure_of_merit + ".npy").is_file()
            and path.joinpath("scores_list_" + figure_of_merit + ".npy").is_file()
        ):
            training_data = np.load(str(path / ("training_data_" + figure_of_merit + ".npy")), allow_pickle=True)
            names_list = list(np.load(str(path / ("names_list_" + figure_of_merit + ".npy")), allow_pickle=True))
            scores_list = list(np.load(str(path / ("scores_list_" + figure_of_merit + ".npy")), allow_pickle=True))
        else:
            error_msg = "Training data not found. Please run the training script first."
            raise FileNotFoundError(error_msg)

        return training_data, names_list, scores_list
