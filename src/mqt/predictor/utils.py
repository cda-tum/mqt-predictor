import json
import signal
import sys

if sys.version_info < (3, 10, 0):
    import importlib_resources as resources
else:
    from importlib import resources

from pathlib import Path

import numpy as np
from joblib import dump
from qiskit import QuantumCircuit
from qiskit.test.mock.backends import FakeMontreal, FakeWashington


def get_width_penalty():
    """Returns the penalty value if a quantum computer has not enough qubits."""
    width_penalty = -10000
    return width_penalty


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


def get_compilation_pipeline():
    compilation_pipeline = {
        "devices": {
            "ibm": [("ibm_washington", 127), ("ibm_montreal", 27)],
            "rigetti": [("rigetti_aspen_m1", 80)],
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


def dict_to_featurevector(gate_dict):
    """Calculates and returns the feature vector of a given quantum circuit gate dictionary."""
    res_dct = dict.fromkeys(get_openqasm_gates(), 0)
    for key, val in dict(gate_dict).items():
        if key in res_dct:
            res_dct[key] = val

    return res_dct


def timeout_watcher(func, args, timeout):
    """Method that stops a function call after a given timeout limit."""

    class TimeoutException(Exception):  # Custom exception class
        pass

    def timeout_handler(signum, frame):  # Custom signal handler
        raise TimeoutException

    # Change the behavior of SIGALRM
    signal.signal(signal.SIGALRM, timeout_handler)

    signal.alarm(timeout)
    try:
        res = func(*args)
    except TimeoutException:
        print("Calculation/Generation exceeded timeout limit for ", func, args[1:])
        return False
    except Exception as e:
        print("Something else went wrong: ", e)
        return False
    else:
        # Reset the alarm
        signal.alarm(0)

    return res


def calc_eval_score_for_qc(qc_path: str, device: str):
    try:
        qc = QuantumCircuit.from_qasm_file(qc_path)
    except Exception as e:
        print("Fail in calc_eval_score_for_qc: ", e)
        return get_width_penalty()
    res = 1

    if "ibm_montreal" in device or "ibm_washington" in device:

        if "ibm_montreal" in device:
            backend = ibm_montreal_calibration
        else:
            backend = ibm_washington_calibration
        for instruction, qargs, _cargs in qc.data:
            gate_type = instruction.name
            qubit_indices = [elem.index for elem in qargs]

            assert gate_type in ["rz", "sx", "x", "cx", "measure", "barrier"]

            if gate_type != "barrier":
                assert len(qubit_indices) in [1, 2]

                first_qubit = int(qubit_indices[0])
                if len(qubit_indices) == 1 and gate_type != "measure":
                    specific_error = backend.gate_error(gate_type, [first_qubit])
                elif len(qubit_indices) == 1 and gate_type == "measure":
                    specific_error = backend.readout_error(first_qubit)
                elif len(qubit_indices) == 2:
                    second_qubit = int(qubit_indices[1])
                    specific_error = backend.gate_error(
                        gate_type, [first_qubit, second_qubit]
                    )

                res *= 1 - float(specific_error)

    elif "oqc_lucy" in device:
        for instruction, qargs, _cargs in qc.data:
            gate_type = instruction.name
            qubit_indices = [elem.index for elem in qargs]

            assert gate_type in ["rz", "sx", "x", "ecr", "measure", "barrier"]
            if gate_type != "barrier":
                assert len(qubit_indices) in [1, 2]

                first_qubit = int(qubit_indices[0])
                if len(qubit_indices) == 1 and gate_type != "measure":
                    specific_fidelity = oqc_lucy_calibration["fid_1Q"][str(first_qubit)]
                elif len(qubit_indices) == 1 and gate_type == "measure":
                    specific_fidelity = oqc_lucy_calibration["fid_1Q_readout"][
                        str(first_qubit)
                    ]
                elif len(qubit_indices) == 2:
                    second_qubit = int(qubit_indices[1])
                    tmp = str(first_qubit) + "-" + str(second_qubit)
                    if oqc_lucy_calibration["fid_2Q"].get(tmp) is None:
                        specific_fidelity = oqc_lucy_calibration["avg_2Q"]
                    else:
                        specific_fidelity = oqc_lucy_calibration["fid_2Q"][tmp]

                res *= specific_fidelity
    elif "rigetti_aspen_m1" in device:
        mapping = get_rigetti_qubit_dict()
        for instruction, qargs, _cargs in qc.data:
            gate_type = instruction.name
            qubit_indices = [elem.index for elem in qargs]

            assert gate_type in ["rx", "rz", "cz", "measure", "barrier"]
            if gate_type != "barrier":
                assert len(qubit_indices) in [1, 2]

                first_qubit = int(qubit_indices[0])
                if len(qubit_indices) == 1 and gate_type in ["rx", "rz", "cz"]:
                    specific_fidelity = rigetti_m1_calibration["fid_1Q"][
                        mapping.get(str(first_qubit))
                    ]
                elif len(qubit_indices) == 1 and gate_type == "measure":
                    specific_fidelity = rigetti_m1_calibration["fid_1Q_readout"][
                        mapping.get(str(first_qubit))
                    ]
                elif len(qubit_indices) == 2:
                    second_qubit = int(qubit_indices[1])
                    tmp = (
                        str(
                            min(
                                int(mapping.get(str(first_qubit))),
                                int(mapping.get(str(second_qubit))),
                            )
                        )
                        + "-"
                        + str(
                            max(
                                int(mapping.get(str(first_qubit))),
                                int(mapping.get(str(second_qubit))),
                            )
                        )
                    )
                    if (
                        rigetti_m1_calibration["fid_2Q_CZ"].get(tmp) is None
                        or rigetti_m1_calibration["fid_2Q_CZ"][tmp] is None
                    ):
                        specific_fidelity = rigetti_m1_calibration["avg_2Q"]
                    else:
                        specific_fidelity = rigetti_m1_calibration["fid_2Q_CZ"][tmp]

                res *= specific_fidelity

    elif "ionq11" in device:
        for instruction, qargs, _cargs in qc.data:
            gate_type = instruction.name
            qubit_indices = [elem.index for elem in qargs]

            assert gate_type in ["rxx", "rz", "ry", "rx", "measure", "barrier"]
            if gate_type != "barrier":
                assert len(qubit_indices) in [1, 2]

                if len(qubit_indices) == 1:
                    specific_fidelity = ionq_calibration["avg_1Q"]
                elif len(qubit_indices) == 2:
                    specific_fidelity = ionq_calibration["avg_2Q"]
                res *= specific_fidelity
    else:
        print("Error: No suitable backend found!")

    return res


def init_all_config_files():
    try:
        global ibm_montreal_calibration
        ibm_montreal_calibration = FakeMontreal().properties()
        global ibm_washington_calibration
        ibm_washington_calibration = FakeWashington().properties()
        global oqc_lucy_calibration
        oqc_lucy_calibration = parse_oqc_calibration_config()
        global rigetti_m1_calibration
        rigetti_m1_calibration = parse_rigetti_calibration_config()
        global ionq_calibration
        ionq_calibration = parse_ionq_calibration_config()

    except Exception as e:
        print("init_all_config_files() failed: ", e)
        return False
    else:
        return True


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
    ) = calc_supermarq_features(qc)
    feature_dict["program_communication"] = program_communication
    feature_dict["critical_depth"] = critical_depth
    feature_dict["entanglement_ratio"] = entanglement_ratio
    feature_dict["parallelism"] = parallelism
    feature_dict["liveness"] = liveness

    return feature_dict


def get_rigetti_qubit_dict():
    mapping = {
        "32": "4",
        "39": "3",
        "38": "2",
        "37": "1",
        "36": "0",
        "35": "7",
        "34": "6",
        "33": "5",
        "25": "15",
        "24": "14",
        "31": "13",
        "30": "12",
        "29": "11",
        "28": "10",
        "27": "17",
        "26": "16",
        "17": "25",
        "16": "24",
        "23": "23",
        "22": "22",
        "21": "21",
        "20": "20",
        "19": "27",
        "18": "26",
        "8": "34",
        "9": "35",
        "10": "36",
        "11": "37",
        "12": "30",
        "13": "31",
        "14": "32",
        "15": "33",
        "0": "44",
        "1": "45",
        "2": "46",
        "3": "47",
        "4": "40",
        "5": "41",
        "6": "42",
        "7": "43",
        "72": "104",
        "73": "105",
        "74": "106",
        "75": "107",
        "76": "100",
        "77": "101",
        "78": "102",
        "79": "103",
        "64": "114",
        "65": "115",
        "66": "116",
        "67": "117",
        "68": "110",
        "69": "111",
        "70": "112",
        "71": "113",
        "56": "124",
        "57": "125",
        "58": "126",
        "59": "127",
        "60": "120",
        "61": "121",
        "62": "122",
        "63": "123",
        "48": "134",
        "49": "135",
        "50": "136",
        "51": "137",
        "52": "130",
        "53": "131",
        "54": "132",
        "55": "133",
        "40": "144",
        "41": "145",
        "42": "146",
        "43": "147",
        "44": "140",
        "45": "141",
        "46": "142",
        "47": "143",
    }

    assert len(mapping) == 80
    return mapping


def parse_ionq_calibration_config():
    ref = (
        resources.files("mqt.predictor") / "calibration_files" / "ionq_calibration.json"
    )
    with ref.open() as f:
        ionq_calibration = json.load(f)
    ionq_dict = {
        "backend": "ionq",
        "avg_1Q": ionq_calibration["fidelity"]["1Q"].get("mean"),
        "avg_2Q": ionq_calibration["fidelity"]["2Q"].get("mean"),
    }
    return ionq_dict


def parse_oqc_calibration_config():
    ref = (
        resources.files("mqt.predictor")
        / "calibration_files"
        / "oqc_lucy_calibration.json"
    )
    with ref.open() as f:
        oqc_lucy_calibration = json.load(f)
    fid_1Q = {}
    fid_1Q_readout = {}
    for elem in oqc_lucy_calibration["oneQubitProperties"]:
        fid_1Q[str(elem)] = oqc_lucy_calibration["oneQubitProperties"][elem][
            "oneQubitFidelity"
        ][0].get("fidelity")
        fid_1Q_readout[str(elem)] = oqc_lucy_calibration["oneQubitProperties"][elem][
            "oneQubitFidelity"
        ][1].get("fidelity")
    fid_2Q = {}
    for elem in oqc_lucy_calibration["twoQubitProperties"]:
        fid_2Q[str(elem)] = oqc_lucy_calibration["twoQubitProperties"][elem][
            "twoQubitGateFidelity"
        ][0].get("fidelity")

    avg_1Q = np.average(list(fid_1Q.values()))
    avg_2Q = np.average(list(fid_2Q.values()))
    oqc_dict = {
        "backend": "oqc_lucy",
        "avg_1Q": avg_1Q,
        "fid_1Q": fid_1Q,
        "fid_1Q_readout": fid_1Q_readout,
        "avg_2Q": avg_2Q,
        "fid_2Q": fid_2Q,
    }
    return oqc_dict


def parse_rigetti_calibration_config():
    ref = (
        resources.files("mqt.predictor")
        / "calibration_files"
        / "rigetti_m1_calibration.json"
    )
    with ref.open() as f:
        rigetti_m1_calibration = json.load(f)
    fid_1Q = {}
    fid_1Q_readout = {}
    missing_indices = []
    for elem in rigetti_m1_calibration["specs"]["1Q"]:

        fid_1Q[str(elem)] = rigetti_m1_calibration["specs"]["1Q"][elem].get("f1QRB")
        fid_1Q_readout[str(elem)] = rigetti_m1_calibration["specs"]["1Q"][elem].get(
            "fRO"
        )

    fid_2Q_CZ = {}
    non_list = []
    for elem in rigetti_m1_calibration["specs"]["2Q"]:
        if rigetti_m1_calibration["specs"]["2Q"][elem].get("fCZ") is None:
            non_list.append(elem)
        else:
            fid_2Q_CZ[str(elem)] = rigetti_m1_calibration["specs"]["2Q"][elem].get(
                "fCZ"
            )

    cz_fid_avg = np.average(list(fid_2Q_CZ.values()))

    avg_1Q = np.average(list(fid_1Q.values()))
    for elem in missing_indices:
        fid_2Q_CZ[elem] = cz_fid_avg

    rigetti_dict = {
        "backend": "rigetti_aspen_m1",
        "avg_1Q": avg_1Q,
        "fid_1Q": fid_1Q,
        "fid_1Q_readout": fid_1Q_readout,
        "avg_2Q": cz_fid_avg,
        "fid_2Q_CZ": fid_2Q_CZ,
    }
    return rigetti_dict


def calc_supermarq_features(qc: QuantumCircuit):

    connectivity = []
    liveness_A_matrix = 0
    for _ in range(qc.num_qubits):
        connectivity.append([])

    offset = 0
    qubit_indices = []
    for elem in qc.qregs:
        for i in range(elem.size):
            qubit_indices.append(offset + i)
        offset += elem.size

    for instruction, _, _ in qc.data:
        gate_type = instruction.name
        liveness_A_matrix += len(qubit_indices)
        if gate_type != "barrier":
            all_indices = set(qubit_indices)
            for qubit_index in all_indices:
                to_be_added_entries = all_indices.copy()
                to_be_added_entries.remove(int(qubit_index))
                connectivity[int(qubit_index)].extend(to_be_added_entries)

    for i in range(qc.num_qubits):
        connectivity[i] = len(set(connectivity[i]))

    num_gates = sum(qc.count_ops().values())
    num_multiple_qubit_gates = qc.num_nonlocal_gates()
    depth = qc.depth()
    program_communication = np.sum(connectivity) / (qc.num_qubits * (qc.num_qubits - 1))

    if num_multiple_qubit_gates == 0:
        critical_depth = 0
    else:
        critical_depth = (
            qc.depth(filter_function=lambda x: len(x[1]) > 1) / num_multiple_qubit_gates
        )

    entanglement_ratio = num_multiple_qubit_gates / num_gates
    assert num_multiple_qubit_gates <= num_gates

    parallelism = (num_gates / depth - 1) / (qc.num_qubits - 1)

    liveness = liveness_A_matrix / (depth * qc.num_qubits)

    return (
        program_communication,
        critical_depth,
        entanglement_ratio,
        parallelism,
        liveness,
    )


def postprocess_ocr_qasm_files(directory: str = None):
    if directory is None:
        directory = str(
            resources.files("mqt.predictor").joinpath("training_samples_compiled")
        )

    for filename in Path(directory).iterdir():
        filename = str(filename).split("/")[-1]
        if "qasm" in filename:
            comp_path_index = int(filename.split("_")[-1].split(".")[0])
            filepath = str(Path(directory) / filename)
            # checking if it is a file
            if comp_path_index >= 24 and comp_path_index <= 27:
                with open(filepath) as f:
                    lines = f.readlines()
                with open(filepath, "w") as f:
                    for line in lines:
                        if not (
                            "gate rzx" in line.strip("\n")
                            or "gate ecr" in line.strip("\n")
                        ):
                            f.write(line)
                        if "gate ecr" in line.strip("\n"):
                            f.write(
                                "gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }\n"
                            )
                            f.write(
                                "gate ecr q0,q1 { rzx(pi/4) q0,q1; x q0; rzx(-pi/4) q0,q1; }\n"
                            )

                print("New qasm file for: ", filepath)

            elif comp_path_index >= 28 and comp_path_index <= 29:
                with open(filepath) as f:
                    lines = f.readlines()
                with open(filepath, "w") as f:
                    for count, line in enumerate(lines):
                        f.write(line)
                        if count == 9:
                            f.write(
                                "gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }\n"
                            )
                            f.write(
                                "gate ecr q0,q1 { rzx(pi/4) q0,q1; x q0; rzx(-pi/4) q0,q1; }\n"
                            )
                print("New qasm file for: ", filepath)


def save_classifier(clf):
    dump(clf, "trained_clf.joblib")


def save_training_data(res):
    training_data, names_list, scores_list = res

    with resources.as_file(resources.files("mqt.predictor") / "training_data") as path:
        data = np.asarray(training_data)
        np.save(str(path / "training_data.npy"), data)
        data = np.asarray(names_list)
        np.save(str(path / "names_list.npy"), data)
        data = np.asarray(scores_list)
        np.save(str(path / "scores_list.npy"), data)


def load_training_data():
    with resources.as_file(resources.files("mqt.predictor") / "training_data") as path:
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
