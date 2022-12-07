import json
import signal
import sys

if sys.version_info < (3, 10, 0):
    import importlib_resources as resources
else:
    from importlib import resources

import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import FakeMontreal, FakeWashington


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


def reward_crit_depth(qc):
    (
        program_communication,
        critical_depth,
        entanglement_ratio,
        parallelism,
        liveness,
    ) = calc_supermarq_features(qc)
    return 1 - critical_depth


def reward_parallelism(qc):
    (
        program_communication,
        critical_depth,
        entanglement_ratio,
        parallelism,
        liveness,
    ) = calc_supermarq_features(qc)
    return critical_depth


def reward_expected_fidelity(qc_or_path: str, device: str):
    if isinstance(qc_or_path, QuantumCircuit):
        qc = qc_or_path
    else:
        try:
            qc = QuantumCircuit.from_qasm_file(qc_or_path)
        except Exception as e:
            print("Fail in reward_expected_fidelity reading a the quantum circuit: ", e)
            return 0
    res = 1

    if "ibm_montreal" in device or "ibm_washington" in device:

        if "ibm_montreal" in device:
            backend = ibm_montreal_calibration
        else:
            backend = ibm_washington_calibration

        for instruction, qargs, _cargs in qc.data:
            gate_type = instruction.name

            assert gate_type in ["rz", "sx", "x", "cx", "measure", "barrier"]

            if gate_type != "barrier":
                assert len(qargs) in [1, 2]
                first_qubit = calc_qubit_index(qargs, qc.qregs, 0)
                if len(qargs) == 1:
                    try:
                        if gate_type == "measure":
                            specific_error = backend.readout_error(first_qubit)
                        else:
                            specific_error = backend.gate_error(
                                gate_type, [first_qubit]
                            )
                    except Exception as e:
                        print(instruction, qargs)
                        print(
                            "Error in IBM backend.gate_error(): ",
                            e,
                            device,
                            first_qubit,
                        )
                        return 0
                else:
                    second_qubit = calc_qubit_index(qargs, qc.qregs, 1)
                    try:
                        specific_error = backend.gate_error(
                            gate_type, [first_qubit, second_qubit]
                        )
                        if specific_error == 1:
                            specific_error = ibm_washington_cx_mean_error
                    except Exception as e:
                        print(instruction, qargs)
                        print(
                            "Error in IBM backend.gate_error(): ",
                            e,
                            device,
                            first_qubit,
                            second_qubit,
                        )
                        return 0

                res *= 1 - float(specific_error)
    elif "oqc_lucy" in device:
        for instruction, qargs, _cargs in qc.data:
            gate_type = instruction.name

            assert gate_type in ["rz", "sx", "x", "ecr", "measure", "barrier"]
            if gate_type != "barrier":
                assert len(qargs) in [1, 2]
                first_qubit = calc_qubit_index(qargs, qc.qregs, 0)
                if len(qargs) == 1 and gate_type != "measure":
                    specific_fidelity = oqc_lucy_calibration["fid_1Q"][str(first_qubit)]
                elif len(qargs) == 1 and gate_type == "measure":
                    specific_fidelity = oqc_lucy_calibration["fid_1Q_readout"][
                        str(first_qubit)
                    ]
                elif len(qargs) == 2:
                    second_qubit = calc_qubit_index(qargs, qc.qregs, 1)
                    tmp = str(first_qubit) + "-" + str(second_qubit)
                    if oqc_lucy_calibration["fid_2Q"].get(tmp) is None:
                        specific_fidelity = oqc_lucy_calibration["avg_2Q"]
                    else:
                        specific_fidelity = oqc_lucy_calibration["fid_2Q"][tmp]

                res *= specific_fidelity

    elif "ionq11" in device:
        for instruction, qargs, _cargs in qc.data:
            gate_type = instruction.name

            assert gate_type in ["rxx", "rz", "ry", "rx", "measure", "barrier"]
            if gate_type != "barrier":
                assert len(qargs) in [1, 2]

                if len(qargs) == 1:
                    specific_fidelity = ionq_calibration["avg_1Q"]
                elif len(qargs) == 2:
                    specific_fidelity = ionq_calibration["avg_2Q"]
                res *= specific_fidelity
    elif "rigetti_aspen_m2" in device:

        mapping = get_rigetti_qubit_dict()
        for instruction, qargs, _cargs in qc.data:
            gate_type = instruction.name

            assert gate_type in ["rx", "rz", "cz", "measure", "barrier"]
            if gate_type != "barrier":
                assert len(qargs) in [1, 2]
                first_qubit = calc_qubit_index(qargs, qc.qregs, 0)
                if len(qargs) == 1:
                    if gate_type == "measure":
                        specific_fidelity = rigetti_m2_calibration["fid_1Q_readout"][
                            mapping.get(str(first_qubit))
                        ]
                    else:
                        specific_fidelity = rigetti_m2_calibration["fid_1Q"][
                            mapping.get(str(first_qubit))
                        ]
                else:
                    second_qubit = calc_qubit_index(qargs, qc.qregs, 1)
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
                        rigetti_m2_calibration["fid_2Q_CZ"].get(tmp) is None
                        or rigetti_m2_calibration["fid_2Q_CZ"][tmp] is None
                    ):
                        specific_fidelity = rigetti_m2_calibration["avg_2Q"]
                    else:
                        specific_fidelity = rigetti_m2_calibration["fid_2Q_CZ"][tmp]

                res *= specific_fidelity

    else:
        print("Error: No suitable backend found!")

    return res


def calc_qubit_index(qargs, qregs, index):
    offset = 0
    for reg in qregs:
        if qargs[index] not in reg:
            offset += reg.size
        else:
            qubit_index = offset + reg.index(qargs[index])
            return qubit_index
    raise ValueError("Qubit not found.")


def init_all_config_files():
    try:
        global ibm_washington_cx_mean_error
        ibm_washington_cx_mean_error = get_mean_IBM_washington_cx_error()
        global ibm_montreal_calibration
        ibm_montreal_calibration = FakeMontreal().properties()
        global ibm_washington_calibration
        ibm_washington_calibration = FakeWashington().properties()
        global oqc_lucy_calibration
        oqc_lucy_calibration = parse_oqc_calibration_config()
        global rigetti_m2_calibration
        rigetti_m2_calibration = parse_rigetti_calibration_config()
        global ionq_calibration
        ionq_calibration = parse_ionq_calibration_config()

    except Exception as e:
        print("init_all_config_files() failed: ", e)
        return False
    else:
        return True


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
    for elem in rigetti_m2_calibration["specs"]["2Q"]:
        if rigetti_m2_calibration["specs"]["2Q"][elem].get("fCZ") is None:
            non_list.append(elem)
        else:
            fid_2Q_CZ[str(elem)] = rigetti_m2_calibration["specs"]["2Q"][elem].get(
                "fCZ"
            )

    cz_fid_avg = np.average(list(fid_2Q_CZ.values()))

    avg_1Q = np.average(list(fid_1Q.values()))
    for elem in missing_indices:
        fid_2Q_CZ[elem] = cz_fid_avg

    rigetti_dict = {
        "backend": "rigetti_aspen_m2",
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


def get_mean_IBM_washington_cx_error():
    cmap = FakeWashington().configuration().coupling_map
    backend = FakeWashington().properties()
    somelist = [x for x in cmap if backend.gate_error("cx", x) < 1]

    res = []
    for elem in somelist:
        res.append(backend.gate_error("cx", elem))
    import numpy as np

    mean_cx_error = np.mean(res)
    return mean_cx_error
