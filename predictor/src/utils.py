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


def get_cmap_oqc_lucy():
    """Returns the coupling map of the OQC Lucy quantum computer."""
    # source: https://github.com/aws/amazon-braket-examples/blob/main/examples/braket_features/Verbatim_Compilation.ipynb

    # Connections are NOT bidirectional, this is not an accident
    c_map_oqc_lucy = [[0, 1], [0, 7], [1, 2], [2, 3], [7, 6], [6, 5], [4, 3], [4, 5]]

    return c_map_oqc_lucy


def get_cmap_rigetti_m1():
    """Returns a coupling map of the circular layout scheme used by Rigetti.

    Keyword arguments:
    circles -- number of circles, each one comprises 8 qubits
    """
    c_map_rigetti = []
    for j in range(5):
        for i in range(0, 7):
            c_map_rigetti.append([i + j * 8, i + 1 + j * 8])

            if i == 6:
                c_map_rigetti.append([0 + j * 8, 7 + j * 8])

        if j != 0:
            c_map_rigetti.append([j * 8 - 6, j * 8 + 5])
            c_map_rigetti.append([j * 8 - 7, j * 8 + 6])

    for j in range(5):
        m = 8 * j + 5 * 8
        for i in range(0, 7):
            c_map_rigetti.append([i + m, i + 1 + m])

            if i == 6:
                c_map_rigetti.append([0 + m, 7 + m])

        if j != 0:
            c_map_rigetti.append([m - 6, m + 5])
            c_map_rigetti.append([m - 7, m + 6])

    for n in range(5):
        c_map_rigetti.append([n * 8 + 3, n * 8 + 5 * 8])
        c_map_rigetti.append([n * 8 + 4, n * 8 + 7 + 5 * 8])

    inverted = [[item[1], item[0]] for item in c_map_rigetti]
    c_map_rigetti = c_map_rigetti + inverted

    return c_map_rigetti


def get_rigetti_m1():
    """Returns the backend information of the Rigetti M1 Aspen Quantum Computer."""
    rigetti_m1 = {
        "provider": "rigetti",
        "name": "m1",
        "num_qubits": 80,
        "t1_avg": 33.845,
        "t2_avg": 28.230,
        "avg_gate_time_1q": 60e-3,  # source: https://qcs.rigetti.com/qpus -> ASPEN-M-1
        "avg_gate_time_2q": 160e-3,  # source: https://qcs.rigetti.com/qpus -> ASPEN-M-1
        "fid_1q": get_rigetti_m1_fid1(),  # calculated by myself based on data sheet from aws
        "fid_2q": get_rigetti_m1_fid2(),  # calculated by myself based on data sheet from aws
    }
    return rigetti_m1


def get_ibm_washington():
    """Returns the backend information of the IBM Washington Quantum Computer."""
    ibm_washington = {
        "provider": "ibm",
        "name": "washington",
        "num_qubits": 127,
        "t1_avg": 103.39,
        "t2_avg": 97.75,
        "avg_gate_time_1q": 159.8e-3,  # estimated, based on the rigetti relation between 1q and 2q and given avg 2q time
        "avg_gate_time_2q": 550.41e-3,  # source: https://quantum-computing.ibm.com/services?services=systems&system=ibm_washington
        "fid_1q": 0.998401,  # from IBMQ website
        "fid_2q": 0.95439,  # from IBMQ website
    }
    return ibm_washington


def get_ibm_montreal():
    """Returns the backend information of the IBM Montreal Quantum Computer."""
    ibm_montreal = {
        "provider": "ibm",
        "name": "Montreal",
        "num_qubits": 27,
        "t1_avg": 120.55,
        "t2_avg": 74.16,
        "avg_gate_time_1q": 206e-3,  # estimated, based on the rigetti relation between 1q and 2q and given avg 2q time
        "avg_gate_time_2q": 426.159e-3,  # source: https://quantum-computing.ibm.com/services?services=systems&system=ibm_montreal
        "fid_1q": 0.9994951,  # from IBMQ website
        "fid_2q": 0.98129,  # from IBMQ website
    }
    return ibm_montreal


def get_ionq():
    """Returns the backend information of the 11-qubit IonQ Quantum Computer."""
    ionq = {
        "provider": "ionq",
        "name": "IonQ",
        "num_qubits": 11,
        "t1_avg": 10000,
        "t2_avg": 0.2,
        "avg_gate_time_1q": 0.00001,
        "avg_gate_time_2q": 0.0002,
        "fid_1q": 0.9963,  # from AWS
        "fid_2q": 0.9581,  # from AWS
    }
    return ionq


def get_oqc_lucy():
    """Returns the backend information of the OQC Lucy Quantum Computer."""
    oqc_lucy = {
        "provider": "oqc",
        "name": "Lucy",
        "num_qubits": 8,
        "t1_avg": 34.2375,
        "t2_avg": 49.2875,
        "avg_gate_time_1q": 60e-3,  # copied from Rigetti Aspen, number is NOT official from OQC itself
        "avg_gate_time_2q": 160e-3,  # copied from Rigetti Aspen, number is NOT official from OQC itself
        "fid_1q": 0.99905,  # from AWS averaged by myself
        "fid_2q": 0.9375,  # from AWS averaged by myself
    }
    return oqc_lucy


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


def get_machines():
    machines = [
        "qiskit_ionq_opt2",
        "qiskit_ibm_washington_opt2",
        "qiskit_ibm_montreal_opt2",
        "qiskit_rigetti_opt2",
        "qiskit_oqc_opt2",
        "qiskit_ionq_opt3",
        "qiskit_ibm_washington_opt3",
        "qiskit_ibm_montreal_opt3",
        "qiskit_rigetti_opt3",
        "qiskit_oqc_opt3",
        "tket_ionq",
        "tket_ibm_washington_line",
        "tket_ibm_montreal_line",
        "tket_rigetti_line",
        "tket_oqc_line",
        "tket_ibm_washington_graph",
        "tket_ibm_montreal_graph",
        "tket_rigetti_graph",
        "tket_oqc_graph",
    ]
    return machines


def get_rigetti_m1_fid1():
    """Calculates and returns the single gate fidelity for the Rigetti M1."""
    f = open("rigetti_m1_calibration.json")
    rigetti_json = json.load(f)

    fid1 = []
    for elem in rigetti_json["specs"]["1Q"]:
        # for elem2 in (rigetti_json["specs"]["1Q"][elem]):
        fid1.append(rigetti_json["specs"]["1Q"][elem]["f1QRB"])
    avg_fid1 = sum(fid1) / len(fid1)

    return avg_fid1


def get_rigetti_m1_fid2():
    """Calculates and returns the two gate fidelity for the Rigetti M1."""
    f = open("rigetti_m1_calibration.json")
    rigetti_json = json.load(f)
    fid2 = []
    for elem in rigetti_json["specs"]["2Q"]:
        val = rigetti_json["specs"]["2Q"][elem].get("fCZ")
        if val:
            fid2.append(val)
    avg_fid2 = sum(fid2) / len(fid2)
    return avg_fid2


def get_oqc_lucy_fid2():
    """Calculates and returns the avg two gate fidelity for the OQC Lucy."""
    with open("oqc_lucy_calibration.json", "r") as f:
        backend = json.load(f)

    fid2 = []
    for elem in backend["twoQubitProperties"]:
        val = backend["twoQubitProperties"][elem]["twoQubitGateFidelity"][0].get(
            "fidelity"
        )
        if val:
            fid2.append(val)

    avg_fid2 = sum(fid2) / len(fid2)
    return avg_fid2


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
        return None
    except Exception as e:
        print("Something else went wrong: ", e)
        return None
    else:
        # Reset the alarm
        signal.alarm(0)

    return res


def get_compiled_output_folder():
    return "qasm_compiled/"


def calc_eval_score_for_qc(qc_path):
    # read qasm to Qiskit Quantumcircuit
    try:
        qc = QuantumCircuit.from_qasm_file(qc_path)
    except Exception as e:
        print("Fail in calc_eval_score_for_qc: ", e)
        return get_width_penalty()
    res = 1

    if "ibm_montreal" in qc_path or "ibm_washington" in qc_path:

        if "ibm_montreal" in qc_path:
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

    elif "oqc" in qc_path:
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
    elif "rigetti" in qc_path:
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

    elif "ionq" in qc_path:
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

    print("Eval score for :", qc_path, " is ", res)
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


def postprocess_ocr_qasm_files(directory: str = "qasm_compiled"):

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if "oqc_qiskit" in f:
            with open(f, "r") as f:
                lines = f.readlines()
            new_name = os.path.join("qasm_compiled_postprocessed", filename)
            with open(new_name, "w") as f:
                for line in lines:
                    if not (
                        "gate rzx" in line.strip("\n") or "gate ecr" in line.strip("\n")
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

        elif "oqc_tket" in f:
            with open(f, "r") as f:
                lines = f.readlines()
            new_name = os.path.join("qasm_compiled_postprocessed", filename)
            with open(new_name, "w") as f:
                count = 0
                for line in lines:
                    f.write(line)
                    count += 1
                    if count == 2:
                        f.write(
                            "gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }\n"
                        )
                        f.write(
                            "gate ecr q0,q1 { rzx(pi/4) q0,q1; x q0; rzx(-pi/4) q0,q1; }\n"
                        )
            qc = QuantumCircuit.from_qasm_file(new_name)
            print("New qasm file for: ", new_name)


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
