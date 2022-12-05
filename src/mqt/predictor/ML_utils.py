import sys

if sys.version_info < (3, 10, 0):
    import importlib_resources as resources
else:
    from importlib import resources

from pathlib import Path

import numpy as np
from joblib import dump
from qiskit import QuantumCircuit

from mqt.predictor import utils


def get_path_trained_model_ML():
    return resources.files("mqt.predictor") / "trained_model_ML"


def get_path_training_circuits_ML():
    return resources.files("mqt.predictor") / "training_circuits_ML"


def get_path_training_circuits_ML_compiled():
    return resources.files("mqt.predictor") / "training_circuits_ML_compiled"


def get_width_penalty():
    """Returns the penalty value if a quantum computer has not enough qubits."""
    width_penalty = -10000
    return width_penalty


def get_compilation_pipeline():
    compilation_pipeline = {
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
    return compilation_pipeline


def get_index_to_comppath_LUT():
    compilation_pipeline = get_compilation_pipeline()
    index = 0
    index_to_comppath_LUT = {}
    for gate_set_name, devices in compilation_pipeline.get("devices").items():
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


def get_openqasm_gates():
    """Returns a list of all quantum gates within the openQASM 2.0 standard header."""
    # according to https://github.com/Qiskit/qiskit-terra/blob/main/qiskit/qasm/libs/qelib1.inc
    gate_list = [
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
    return gate_list


def dict_to_featurevector(gate_dict):
    """Calculates and returns the feature vector of a given quantum circuit gate dictionary."""
    res_dct = dict.fromkeys(get_openqasm_gates(), 0)
    for key, val in dict(gate_dict).items():
        if key in res_dct:
            res_dct[key] = val

    return res_dct


def create_feature_dict(qasm_str_or_path: str):

    if len(qasm_str_or_path) < 260 and Path(qasm_str_or_path).exists():
        qc = QuantumCircuit.from_qasm_file(qasm_str_or_path)
    elif "OPENQASM" in qasm_str_or_path:
        qc = QuantumCircuit.from_qasm_str(qasm_str_or_path)
    else:
        print("Neither a qasm file path nor a qasm str has been provided.")
        return False

    ops_list = qc.count_ops()
    feature_dict = dict_to_featurevector(ops_list)

    feature_dict["num_qubits"] = qc.num_qubits
    feature_dict["depth"] = qc.depth()

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


def save_classifier(clf):
    dump(clf, "./src/mqt/predictor/trained_model_ML/trained_clf.joblib")


def save_training_data(res):
    training_data, names_list, scores_list = res

    with resources.as_file(
        resources.files("mqt.predictor") / "training_data_ML_aggregated"
    ) as path:
        data = np.asarray(training_data)
        np.save(str(path / "training_data.npy"), data)
        data = np.asarray(names_list)
        np.save(str(path / "names_list.npy"), data)
        data = np.asarray(scores_list)
        np.save(str(path / "scores_list.npy"), data)


def load_training_data():
    with resources.as_file(
        resources.files("mqt.predictor") / "training_data_ML_aggregated"
    ) as path:
        if (
            path.joinpath("training_data.npy").is_file()
            and path.joinpath("names_list.npy").is_file()
            and path.joinpath("scores_list.npy").is_file()
        ):
            training_data = np.load(str(path / "training_data.npy"), allow_pickle=True)
            names_list = list(np.load(str(path / "names_list.npy"), allow_pickle=True))
            scores_list = list(
                np.load(str(path / "scores_list.npy"), allow_pickle=True)
            )
        else:
            print("Training data loading failed.")
            return

        return training_data, names_list, scores_list
