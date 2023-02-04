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
from mqt.predictor import ml, utils
from qiskit import QuantumCircuit

if TYPE_CHECKING:
    from sklearn.ensemble import RandomForestClassifier


def qcompile(qc: QuantumCircuit | str) -> tuple[QuantumCircuit, int]:
    """Returns the compiled quantum circuit which is compiled with the predicted combination of compilation options.

    Keyword arguments:
    qc -- to be compiled quantum circuit or path to a qasm file

    Returns: compiled quantum circuit as Qiskit QuantumCircuit object
    """

    predictor = ml.Predictor()
    prediction = predictor.predict(qc)
    return predictor.compile_as_predicted(qc, prediction)


def get_path_training_data() -> Path:
    return Path(str(resources.files("mqt.predictor"))) / "ml" / "training_data"


def get_path_trained_model() -> Path:
    return get_path_training_data() / "trained_model"


def get_path_training_circuits() -> Path:
    return get_path_training_data() / "training_circuits"


def get_path_training_circuits_compiled() -> Path:
    return get_path_training_data() / "training_circuits_compiled"


def get_width_penalty() -> int:
    """Returns the penalty value if a quantum computer has not enough qubits."""
    return -10000


def get_compilation_pipeline() -> dict[str, dict[str, Any]]:
    return {
        "devices": {
            "ibm": [("ibm_washington", 127), ("ibm_montreal", 27)],
            "rigetti": [("rigetti_aspen_m2", 80)],
            "ionq": [("ionq11", 11)],
            "oqc": [("oqc_lucy", 8)],
        },
        "compiler": {
            "qiskit": {"optimization_level": [0, 1, 2, 3]},
            "tket": {"lineplacement": [False, True]},
        },
    }


def get_index_to_comppath_LUT() -> dict[int, Any]:
    compilation_pipeline = get_compilation_pipeline()
    index = 0
    index_to_comppath_LUT = {}
    for gate_set_name, devices in compilation_pipeline["devices"].items():
        for device_name, _max_qubits in devices:
            for compiler, settings in compilation_pipeline["compiler"].items():
                if "qiskit" in compiler:
                    for opt_level in settings["optimization_level"]:
                        index_to_comppath_LUT[index] = (
                            gate_set_name,
                            device_name,
                            compiler,
                            opt_level,
                        )
                        index += 1
                elif "tket" in compiler:
                    for lineplacement in settings["lineplacement"]:
                        index_to_comppath_LUT[index] = (
                            gate_set_name,
                            device_name,
                            compiler,
                            lineplacement,
                        )
                        index += 1
    return index_to_comppath_LUT


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


def create_feature_dict(qc: str) -> dict[str, Any]:
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

    (
        program_communication,
        critical_depth,
        entanglement_ratio,
        parallelism,
        liveness,
    ) = utils.calc_supermarq_features(qc)
    feature_dict["program_communication"] = program_communication
    feature_dict["critical_depth"] = critical_depth
    feature_dict["entanglement_ratio"] = entanglement_ratio
    feature_dict["parallelism"] = parallelism
    feature_dict["liveness"] = liveness
    return feature_dict


def save_classifier(clf: RandomForestClassifier) -> None:
    dump(clf, str(get_path_trained_model() / "trained_clf.joblib"))


def save_training_data(res: tuple[list[Any], list[Any], list[Any]]) -> None:
    training_data, names_list, scores_list = res

    with resources.as_file(get_path_training_data() / "training_data_aggregated") as path:
        data = np.asarray(training_data)
        np.save(str(path / "training_data.npy"), data)
        data = np.asarray(names_list)
        np.save(str(path / "names_list.npy"), data)
        data = np.asarray(scores_list)
        np.save(str(path / "scores_list.npy"), data)


def load_training_data() -> tuple[list[Any], list[str], list[Any]]:
    with resources.as_file(get_path_training_data() / "training_data_aggregated") as path:
        if (
            path.joinpath("training_data.npy").is_file()
            and path.joinpath("names_list.npy").is_file()
            and path.joinpath("scores_list.npy").is_file()
        ):
            training_data = np.load(str(path / "training_data.npy"), allow_pickle=True)
            names_list = list(np.load(str(path / "names_list.npy"), allow_pickle=True))
            scores_list = list(np.load(str(path / "scores_list.npy"), allow_pickle=True))
        else:
            error_msg = "Training data not found. Please run the training script first."
            raise FileNotFoundError(error_msg)

        return training_data, names_list, scores_list
