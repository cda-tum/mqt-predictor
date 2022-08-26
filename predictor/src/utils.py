import signal
import json
import numpy as np
from pytket import OpType
from qiskit import QuantumCircuit
import os
from qiskit.test.mock.backends import FakeWashington, FakeMontreal


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
        for device_name, max_qubits in devices:
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
    openqasm_gates_list = get_openqasm_gates()
    res_dct = {openqasm_gates_list[i] for i in range(0, len(openqasm_gates_list))}
    res_dct = dict.fromkeys(res_dct, 0)
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


def get_compiled_output_folder():
    return "qasm_compiled/"


def calc_eval_score_for_qc(qc_path: str, device: str):
    # read qasm to Qiskit Quantumcircuit
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
        for instruction, qargs, cargs in qc.data:
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
        for instruction, qargs, cargs in qc.data:
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
        for instruction, qargs, cargs in qc.data:
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
        for instruction, qargs, cargs in qc.data:
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

    # print("Eval score for :", qc_path, " is ", res)
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


def create_feature_vector(qc_path: str):
    qc = QuantumCircuit.from_qasm_file(qc_path)

    ops_list = qc.count_ops()
    feature_vector = dict_to_featurevector(ops_list)

    feature_vector["num_qubits"] = qc.num_qubits
    feature_vector["depth"] = qc.depth()
    connectivity = calc_connectivity_for_qc(qc)
    for i in range(len(connectivity)):
        feature_vector[str(i + 1) + "_max_interactions"] = connectivity[i]

    return feature_vector


def get_rigetti_qubit_dict():
    mapping = {}
    mapping["32"] = "4"
    mapping["39"] = "3"
    mapping["38"] = "2"
    mapping["37"] = "1"
    mapping["36"] = "0"
    mapping["35"] = "7"
    mapping["34"] = "6"
    mapping["33"] = "5"
    mapping["25"] = "15"
    mapping["24"] = "14"
    mapping["31"] = "13"
    mapping["30"] = "12"
    mapping["29"] = "11"
    mapping["28"] = "10"
    mapping["27"] = "17"
    mapping["26"] = "16"
    mapping["17"] = "25"
    mapping["16"] = "24"
    mapping["23"] = "23"
    mapping["22"] = "22"
    mapping["21"] = "21"
    mapping["20"] = "20"
    mapping["19"] = "27"
    mapping["18"] = "26"
    mapping["8"] = "34"
    mapping["9"] = "35"
    mapping["10"] = "36"
    mapping["11"] = "37"
    mapping["12"] = "30"
    mapping["13"] = "31"
    mapping["14"] = "32"
    mapping["15"] = "33"
    mapping["0"] = "44"
    mapping["1"] = "45"
    mapping["2"] = "46"
    mapping["3"] = "47"
    mapping["4"] = "40"
    mapping["5"] = "41"
    mapping["6"] = "42"
    mapping["7"] = "43"
    mapping["72"] = "104"
    mapping["73"] = "105"
    mapping["74"] = "106"
    mapping["75"] = "107"
    mapping["76"] = "100"
    mapping["77"] = "101"
    mapping["78"] = "102"
    mapping["79"] = "103"
    mapping["64"] = "114"
    mapping["65"] = "115"
    mapping["66"] = "116"
    mapping["67"] = "117"
    mapping["68"] = "110"
    mapping["69"] = "111"
    mapping["70"] = "112"
    mapping["71"] = "113"
    mapping["56"] = "124"
    mapping["57"] = "125"
    mapping["58"] = "126"
    mapping["59"] = "127"
    mapping["60"] = "120"
    mapping["61"] = "121"
    mapping["62"] = "122"
    mapping["63"] = "123"
    mapping["48"] = "134"
    mapping["49"] = "135"
    mapping["50"] = "136"
    mapping["51"] = "137"
    mapping["52"] = "130"
    mapping["53"] = "131"
    mapping["54"] = "132"
    mapping["55"] = "133"
    mapping["40"] = "144"
    mapping["41"] = "145"
    mapping["42"] = "146"
    mapping["43"] = "147"
    mapping["44"] = "140"
    mapping["45"] = "141"
    mapping["46"] = "142"
    mapping["47"] = "143"

    assert len(mapping) == 80
    return mapping


def parse_ionq_calibration_config():
    with open("ionq_calibration.json", "r") as f:
        ionq_calibration = json.load(f)
    ionq_dict = {
        "backend": "ionq",
        "avg_1Q": ionq_calibration["fidelity"]["1Q"].get("mean"),
        "avg_2Q": ionq_calibration["fidelity"]["2Q"].get("mean"),
    }

    return ionq_dict


def parse_oqc_calibration_config():
    with open("oqc_lucy_calibration.json", "r") as f:
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
    oqc_dict
    return oqc_dict


def parse_rigetti_calibration_config():
    with open("rigetti_m1_calibration.json", "r") as f:
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

    import numpy as np

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


def calc_connectivity_for_qc(qc: QuantumCircuit):

    connectivity = []
    for i in range(127):
        connectivity.append([])
    for instruction, qargs, cargs in qc.data:
        gate_type = instruction.name
        qubit_indices = [elem.index for elem in qargs]
        if len(qubit_indices) == 2 and gate_type != "barrier":
            first_qubit = int(qubit_indices[0])
            second_qubit = int(qubit_indices[1])
            connectivity[first_qubit].append(second_qubit)
            connectivity[second_qubit].append(first_qubit)
    for i in range(127):
        connectivity[i] = len(set(connectivity[i]))
    connectivity.sort(reverse=True)
    return connectivity[:5]


def postprocess_ocr_qasm_files(directory: str):
    for filename in os.listdir(directory):
        print(filename)
        if "qasm" in filename:
            comp_path_index = int(filename.split("_")[-1].split(".")[0])
            print("comp_path_index: ", comp_path_index)
            f = os.path.join(directory, filename)
            # checking if it is a file
            if comp_path_index >= 24 and comp_path_index <= 27:
                with open(f, "r") as f:
                    lines = f.readlines()
                new_name = os.path.join(directory, filename)
                with open(new_name, "w") as f:
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

                qc = QuantumCircuit.from_qasm_file(new_name)
                print("New qasm file for: ", new_name)

            elif comp_path_index >= 28 and comp_path_index <= 29:
                with open(f, "r") as f:
                    lines = f.readlines()
                new_name = os.path.join(directory, filename)
                with open(new_name, "w") as f:
                    count = 0
                    for line in lines:
                        f.write(line)
                        count += 1
                        if count == 9:
                            f.write(
                                "gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }\n"
                            )
                            f.write(
                                "gate ecr q0,q1 { rzx(pi/4) q0,q1; x q0; rzx(-pi/4) q0,q1; }\n"
                            )
                qc = QuantumCircuit.from_qasm_file(new_name)
                print("New qasm file for: ", new_name)
