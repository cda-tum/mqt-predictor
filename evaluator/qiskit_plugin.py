from qiskit import transpile
from utils import *
from qiskit.test.mock import FakeMontreal

def get_aqt_gateset():
    from qiskit_aqt_provider import AQTProvider

    aqt = AQTProvider("")
    AQT_backend = aqt.backends.aqt_qasm_simulator
    gateset = AQT_backend.configuration().basis_gates
    return gateset


def get_qiskit_scores(qc, opt_level=0):

    penalty_width = 1000000

    ibm_gates = ["rz", "sx", "x", "cx"]
    rigetti_gates = ["rx", "rz", "cz"]
    ionq_gates = ["rxx", "rz", "ry", "rx"]
    oqc_gates = ["rz", "sx", "x", "ecr"]


    ibm_washington = get_ibm_washington()
    if qc.num_qubits > ibm_washington["num_qubits"]:
        score_ibm_washington = penalty_width
    else:
        qc_ibm = transpile(
            qc,
            basis_gates=ibm_gates,
            optimization_level=opt_level,
            coupling_map=get_cmap_imbq_washington(),
        )
        score_ibm_washington = calc_score(qc_ibm, ibm_washington, "qiskit")

    ibm_montreal = get_ibm_montreal()
    if qc.num_qubits > ibm_montreal["num_qubits"]:
        score_ibm_montreal = penalty_width
    else:
        qc_ibm = transpile(
            qc,
            basis_gates=ibm_gates,
            optimization_level=opt_level,
            coupling_map=FakeMontreal().configuration().coupling_map,
        )
        score_ibm_montreal = calc_score(qc_ibm, ibm_washington, "qiskit")

    ionq = get_ionq()
    if qc.num_qubits > ionq["num_qubits"]:
        score_ionq = penalty_width
    else:
        qc_ion = transpile(qc, basis_gates=ionq_gates, optimization_level=opt_level)
        score_ionq = calc_score(qc_ion, ionq, "qiskit")

    rigetti_m1 = get_rigetti_m1()
    if qc.num_qubits > rigetti_m1["num_qubits"]:
        score_rigetti = penalty_width
    else:
        qc_rigetti = transpile(
            qc,
            basis_gates=rigetti_gates,
            optimization_level=opt_level,
            coupling_map=get_cmap_rigetti_m1(10),
        )
        score_rigetti = calc_score(qc_rigetti, rigetti_m1, "qiskit")

    oqc_lucy = get_oqc_lucy()
    if qc.num_qubits > oqc_lucy["num_qubits"]:
        score_oqc = penalty_width
    else:
        qc_oqc = transpile(
            qc,
            basis_gates=oqc_gates,
            optimization_level=opt_level,
            coupling_map=get_c_map_oqc_lucy(),
        )
        score_oqc = calc_score(qc_oqc, oqc_lucy, "qiskit")

    # print("Scores: ", [score_ibm_washington, score_ionq, score_rigetti])

    print("Scores qiskit: ",
        [score_ibm_washington, score_ibm_montreal, score_ionq, score_rigetti, score_oqc]
    )
    return [
        score_ibm_washington,
        score_ibm_montreal,
        score_ionq,
        score_rigetti,
        score_oqc,
    ]

def get_qiskit_gates(qc, opt_level=0):

    ibm_gates = ["rz", "sx", "x", "cx"]
    rigetti_gates = ["rx", "rz", "cz"]
    ionq_gates = ["rxx", "rz", "ry", "rx"]
    oqc_gates = ["rz", "sx", "x", "ecr"]


    ibm_washington = get_ibm_washington()
    if qc.num_qubits > ibm_washington["num_qubits"]:
        gates_ibm_washington = None
    else:
        qc_ibm = transpile(
            qc,
            basis_gates=ibm_gates,
            optimization_level=opt_level,
            coupling_map=get_cmap_imbq_washington(),
        )
        gates_ibm_washington = count_qubit_gates_ibm(qc_ibm, "ibm")

    ibm_montreal = get_ibm_montreal()
    if qc.num_qubits > ibm_montreal["num_qubits"]:
        gates_ibm_montreal = None
    else:
        qc_ibm = transpile(
            qc,
            basis_gates=ibm_gates,
            optimization_level=opt_level,
            coupling_map=FakeMontreal().configuration().coupling_map,
        )
        gates_ibm_montreal = count_qubit_gates_ibm(qc_ibm, "ibm")

    ionq = get_ionq()
    if qc.num_qubits > ionq["num_qubits"]:
        gates_ionq = None
    else:
        qc_ion = transpile(qc, basis_gates=ionq_gates, optimization_level=opt_level)
        gates_ionq = count_qubit_gates_ibm(qc_ion, "ionq")

    rigetti_m1 = get_rigetti_m1()
    if qc.num_qubits > rigetti_m1["num_qubits"]:
        gates_rigetti = None
    else:
        qc_rigetti = transpile(
            qc,
            basis_gates=rigetti_gates,
            optimization_level=opt_level,
            coupling_map=get_cmap_rigetti_m1(10),
        )
        gates_rigetti = count_qubit_gates_ibm(qc_rigetti, "rigetti")

    oqc_lucy = get_oqc_lucy()
    if qc.num_qubits > oqc_lucy["num_qubits"]:
        gates_oqc = None
    else:
        qc_oqc = transpile(
            qc,
            basis_gates=oqc_gates,
            optimization_level=opt_level,
            coupling_map=get_c_map_oqc_lucy(),
        )
        gates_oqc = count_qubit_gates_ibm(qc_oqc, "oqc")

    # print("Scores: ", [score_ibm_washington, score_ionq, score_rigetti])

    #print("gates qiskit: ",
        #[gates_ibm_washington, gates_ibm_montreal, gates_ionq, gates_rigetti, gates_oqc]
    #)
    return ("qiskit", [
        (gates_ibm_washington, "ibm_washington"),
        (gates_ibm_montreal, "ibm_montreal"),
        (gates_ionq, "ionq"),
        (gates_rigetti, "rigetti_m1"),
        (gates_oqc, "oqc_lucy"),
    ])