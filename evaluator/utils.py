from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from pytket import OpType


def count_qubit_gates_tket(qc, provider: str):
    single_qubit_gates = 0
    two_qubit_gates = 0
    if provider == "ibm":
        # gates: ['id', 'rz', 'sx', 'x', 'cx', 'reset']
        single_qubit_gates += qc.n_gates_of_type(OpType.Rz)
        single_qubit_gates += qc.n_gates_of_type(OpType.SX)
        single_qubit_gates += qc.n_gates_of_type(OpType.X)
        single_qubit_gates += qc.n_gates_of_type(OpType.Rz)
        two_qubit_gates += qc.n_gates_of_type(OpType.CX)

    elif provider == "rigetti":
        # gates: rigetti_native_gates = ["rx", "rz", "cz"]
        single_qubit_gates += qc.n_gates_of_type(OpType.Rx)
        single_qubit_gates += qc.n_gates_of_type(OpType.Rz)
        two_qubit_gates += qc.n_gates_of_type(OpType.CZ)

    elif provider == "ionq":
        # gates: ionq_native_gates = ["ms", "rz", "ry", "rx"] or ["rxx", "rz", "ry", "rx"]
        single_qubit_gates += qc.n_gates_of_type(OpType.Rz)
        single_qubit_gates += qc.n_gates_of_type(OpType.Ry)
        single_qubit_gates += qc.n_gates_of_type(OpType.Rx)
        two_qubit_gates += qc.n_gates_of_type(OpType.XXPhase)

    elif provider == "oqc":
        # gates: oqc_gates = ["rz", "sx", "x", "ecr"]
        single_qubit_gates += qc.n_gates_of_type(OpType.Rz)
        single_qubit_gates += qc.n_gates_of_type(OpType.SX)
        single_qubit_gates += qc.n_gates_of_type(OpType.X)
        two_qubit_gates += qc.n_gates_of_type(OpType.ECR)
    return single_qubit_gates, two_qubit_gates

def count_qubit_gates_ibm(qc, provider: str):
    dag = circuit_to_dag(qc)
    count_gates = dag.count_ops_longest_path()
    single_qubit_gates = 0
    two_qubit_gates = 0
    if provider == "ibm":
        # gates: ['id', 'rz', 'sx', 'x', 'cx', 'reset']
        if "id" in count_gates:
            single_qubit_gates += count_gates["id"]
        if "rz" in count_gates:
            single_qubit_gates += count_gates["rz"]
        if "sx" in count_gates:
            single_qubit_gates += count_gates["sx"]
        if "x" in count_gates:
            single_qubit_gates += count_gates["x"]
        if "cx" in count_gates:
            two_qubit_gates += count_gates["cx"]

    elif provider == "rigetti":
        # gates: rigetti_native_gates = ["rx", "rz", "cz"]
        if "rx" in count_gates:
            single_qubit_gates += count_gates["rx"]
        if "rz" in count_gates:
            single_qubit_gates += count_gates["rz"]
        if "cz" in count_gates:
            two_qubit_gates += count_gates["cz"]

    elif provider == "ionq":
        # gates: ionq_native_gates = ["ms", "rz", "ry", "rx"] or ["rxx", "rz", "ry", "rx"]
        if "rx" in count_gates:
            single_qubit_gates += count_gates["rx"]
        if "ry" in count_gates:
            single_qubit_gates += count_gates["ry"]
        if "rz" in count_gates:
            single_qubit_gates += count_gates["rz"]
        if "rxx" in count_gates:
            two_qubit_gates += count_gates["rxx"]

    elif provider == "oqc":
        # gates: oqc_gates = ["rz", "sx", "x", "ecr"]
        if "rz" in count_gates:
            single_qubit_gates += count_gates["rz"]
        if "sx" in count_gates:
            single_qubit_gates += count_gates["sx"]
        if "x" in count_gates and not "sx" in count_gates:
            single_qubit_gates += count_gates["x"]
        if "ecr" in count_gates:
            two_qubit_gates += count_gates["ecr"]
    return single_qubit_gates, two_qubit_gates


def calc_score_from_str(qc: str, backend):
    qc = QuantumCircuit.from_qasm_str(qc)
    return calc_score(qc, backend)


def calc_score_from_path(filepath, backend):
    qc = QuantumCircuit.from_qasm_file(filepath)
    return calc_score(qc, backend)


def calc_score(qc, backend, compiler):
    if compiler == "qiskit":
        count_gates = count_qubit_gates_ibm(qc, backend["provider"])
    elif compiler == "tket":
        count_gates = count_qubit_gates_tket(qc, backend["provider"])

    penalty_factor_1q = 500
    penalty_factor_2q = 1000

    t_1 = backend["t1_avg"]
    t_2 = backend["t2_avg"]
    avg_gate_time_1q = backend["avg_gate_time_1q"]
    avg_gate_time_2q = backend["avg_gate_time_2q"]
    max_depth_1q = min(t_1, t_2) / avg_gate_time_1q
    max_depth_2q = min(t_1, t_2) / avg_gate_time_2q

    score = (
        count_gates[0] / max_depth_1q * penalty_factor_1q
        + count_gates[1] / max_depth_2q * penalty_factor_2q
    )
    # print("Score: ", score)
    return score


def get_c_map_oqc_lucy():
    """Returns the coupling map of the OQC Lucy quantum computer."""
    # source: https://github.com/aws/amazon-braket-examples/blob/main/examples/braket_features/Verbatim_Compilation.ipynb

    # Connections are NOT bidirectional, this is not an accident
    c_map_oqc_lucy = [[0, 1], [0, 7], [1, 2], [2, 3], [7, 6], [6, 5], [4, 3], [4, 5]]

    return c_map_oqc_lucy


def get_cmap_imbq_washington():
    """Returns the coupling map of the IBM-Q washington quantum computer."""
    c_map_ibmq_washington = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [0, 14],
        [14, 18],
        [18, 19],
        [19, 20],
        [20, 21],
        [21, 22],
        [4, 15],
        [15, 22],
        [22, 23],
        [23, 24],
        [24, 25],
        [25, 26],
        [8, 16],
        [16, 26],
        [26, 27],
        [27, 28],
        [28, 29],
        [29, 30],
        [30, 31],
        [31, 32],
        [9, 10],
        [10, 11],
        [11, 12],
        [12, 13],
        [12, 17],
        [17, 30],
        [32, 36],
        [36, 51],
        [20, 33],
        [33, 39],
        [24, 34],
        [34, 43],
        [28, 35],
        [35, 47],
        [37, 38],
        [38, 39],
        [39, 40],
        [40, 41],
        [41, 42],
        [42, 43],
        [43, 44],
        [44, 45],
        [45, 46],
        [46, 47],
        [47, 48],
        [48, 49],
        [49, 50],
        [50, 51],
        [37, 52],
        [52, 56],
        [41, 53],
        [53, 60],
        [45, 54],
        [54, 64],
        [49, 55],
        [55, 68],
        [56, 57],
        [57, 58],
        [58, 59],
        [59, 60],
        [60, 61],
        [61, 62],
        [62, 63],
        [63, 64],
        [64, 65],
        [65, 66],
        [66, 67],
        [67, 68],
        [68, 69],
        [69, 70],
        [70, 74],
        [74, 89],
        [58, 71],
        [71, 77],
        [62, 72],
        [72, 81],
        [66, 73],
        [73, 85],
        [75, 76],
        [76, 77],
        [77, 78],
        [78, 79],
        [79, 80],
        [80, 81],
        [81, 82],
        [82, 83],
        [83, 84],
        [84, 85],
        [85, 86],
        [86, 87],
        [87, 88],
        [88, 89],
        [75, 90],
        [90, 94],
        [79, 91],
        [91, 98],
        [83, 92],
        [92, 102],
        [87, 93],
        [93, 106],
        [94, 95],
        [95, 96],
        [96, 97],
        [97, 98],
        [98, 99],
        [99, 100],
        [100, 101],
        [101, 102],
        [102, 103],
        [103, 104],
        [104, 105],
        [105, 106],
        [106, 107],
        [107, 108],
        [108, 112],
        [112, 126],
        [96, 109],
        [100, 110],
        [110, 118],
        [104, 111],
        [111, 122],
        [113, 114],
        [114, 115],
        [115, 116],
        [116, 117],
        [117, 118],
        [118, 119],
        [119, 120],
        [120, 121],
        [121, 122],
        [122, 123],
        [123, 124],
        [124, 125],
        [125, 126],
    ]

    inverted = [[item[1], item[0]] for item in c_map_ibmq_washington]
    c_map_ibmq_washington = c_map_ibmq_washington + inverted
    return c_map_ibmq_washington


def get_cmap_rigetti_m1(circles: int = 4):
    """Returns a coupling map of the circular layout scheme used by Rigetti.

    Keyword arguments:
    circles -- number of circles, each one comprises 8 qubits
    """
    if circles != 10:
        c_map_rigetti = []
        for j in range(circles):
            for i in range(0, 7):
                c_map_rigetti.append([i + j * 8, i + 1 + j * 8])

                if i == 6:
                    c_map_rigetti.append([0 + j * 8, 7 + j * 8])

            if j != 0:
                c_map_rigetti.append([j * 8 - 6, j * 8 + 5])
                c_map_rigetti.append([j * 8 - 7, j * 8 + 6])

        inverted = [[item[1], item[0]] for item in c_map_rigetti]
        c_map_rigetti = c_map_rigetti + inverted
    else:
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
    rigetti_m1 = {
        "provider": "rigetti",
        "name": "m1",
        "num_qubits": 80,
        "t1_avg": 33.845,
        "t2_avg": 28.230,
        "avg_gate_time_1q": 60e-3,  # source: https://qcs.rigetti.com/qpus -> ASPEN-M-1
        "avg_gate_time_2q": 160e-3,  # source: https://qcs.rigetti.com/qpus -> ASPEN-M-1
    }
    return rigetti_m1


def get_ibm_washington():
    ibm_washington = {
        "provider": "ibm",
        "name": "washington",
        "num_qubits": 127,
        "t1_avg": 103.39,
        "t2_avg": 97.75,
        "avg_gate_time_1q": 159.8e-3,  # estimated, based on the rigetti relation between 1q and 2q and given avg 2q time
        "avg_gate_time_2q": 550.41e-3,  # source: https://quantum-computing.ibm.com/services?services=systems&system=ibm_washington
    }
    return ibm_washington

def get_ibm_montreal():
    ibm_montreal = {
        "provider": "ibm",
        "name": "Montreal",
        "num_qubits": 27,
        "t1_avg": 120.55,
        "t2_avg": 74.16,
        "avg_gate_time_1q": 206e-3,  # estimated, based on the rigetti relation between 1q and 2q and given avg 2q time
        "avg_gate_time_2q": 426.159e-3,  # source: https://quantum-computing.ibm.com/services?services=systems&system=ibm_montreal
    }
    return ibm_montreal


def get_ionq():
    ionq = {
        "provider": "ionq",
        "name": "IonQ",
        "num_qubits": 11,
        "t1_avg": 10000,
        "t2_avg": 0.2,
        "avg_gate_time_1q": 0.00001,
        "avg_gate_time_2q": 0.0002,
    }
    return ionq


def get_oqc_lucy():
    oqc_lucy = {
        "provider": "oqc",
        "name": "Lucy",
        "num_qubits": 8,
        "t1_avg": 34.2375,
        "t2_avg": 49.2875,
        "avg_gate_time_1q": 60e-3,  # copied from Rigetti Aspen, number is NOT official from OQC itself
        "avg_gate_time_2q": 160e-3,  # copied from Rigetti Aspen, number is NOT official from OQC itself
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
