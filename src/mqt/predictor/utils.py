import logging
import signal

import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler.passes import RemoveBarriers

logger = logging.getLogger("mqtpredictor")


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
        logger.debug(
            "Calculation/Generation exceeded timeout limit for "
            + ", "
            + func
            + ", "
            + args[1:]
        )
        return False
    except Exception as e:
        logger.error("Something else went wrong: " + ", " + e)
        return False
    else:
        # Reset the alarm
        signal.alarm(0)

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


def calc_supermarq_features(qc: QuantumCircuit):
    qc = RemoveBarriers()(qc)
    connectivity = []
    liveness_A_matrix = 0
    for _ in range(qc.num_qubits):
        connectivity.append([])

    for _, qargs, _ in qc.data:
        liveness_A_matrix += len(qargs)
        first_qubit = calc_qubit_index(qargs, qc.qregs, 0)
        all_indices = [first_qubit]
        if len(qargs) == 2:
            second_qubit = calc_qubit_index(qargs, qc.qregs, 1)
            all_indices.append(second_qubit)
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

    assert program_communication >= 0 and program_communication <= 1
    assert critical_depth >= 0 and critical_depth <= 1
    assert entanglement_ratio >= 0 and entanglement_ratio <= 1
    assert parallelism >= 0 and parallelism <= 1
    assert liveness >= 0 and liveness <= 1

    return (
        program_communication,
        critical_depth,
        entanglement_ratio,
        parallelism,
        liveness,
    )
