from qiskit import transpile
from evaluator.utils import *
from qiskit.test.mock import FakeMontreal

def get_aqt_gateset():
    from qiskit_aqt_provider import AQTProvider

    aqt = AQTProvider("")
    AQT_backend = aqt.backends.aqt_qasm_simulator
    gateset = AQT_backend.configuration().basis_gates
    return gateset


def get_qiskit_scores(qasm_qc, opt_level=0):

    penalty_width = 100000

    ibm_gates = ["rz", "sx", "x", "cx"]
    rigetti_gates = ["rx", "rz", "cz"]
    ionq_gates = ["rxx", "rz", "ry", "rx"]
    oqc_gates = ["rz", "sx", "x", "ecr"]

    qc = QuantumCircuit.from_qasm_str(qasm_qc)

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
