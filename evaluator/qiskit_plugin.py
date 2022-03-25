from qiskit import transpile, QuantumCircuit
from evaluator.utils import calc_score_from_str, calc_score, get_cmap_rigetti_m1, get_cmap_imbq_washington, get_ibm_washington, get_rigetti_m1, get_ionq
import numpy as np

def get_aqt_gateset():
    from qiskit_aqt_provider import AQTProvider
    aqt = AQTProvider("")
    AQT_backend = aqt.backends.aqt_qasm_simulator
    gateset = AQT_backend.configuration().basis_gates
    return gateset

def get_qiskit_scores(qasm_qc, opt_level=0):

    ibm_gates = ['id', 'rz', 'sx', 'x', 'cx', 'reset']
    rigetti_gates = ["rx", "rz", "cz"]
    ionq_gates = ["rxx", "rz", "ry", "rx"]

    qc = QuantumCircuit.from_qasm_str(qasm_qc)

    qc_ibm = transpile(qc, basis_gates=ibm_gates, optimization_level=opt_level, coupling_map=get_cmap_imbq_washington())
    ibm_washington = get_ibm_washington()
    score_ibm = calc_score_from_str(qc_ibm.qasm(), ibm_washington)

    ionq = get_ionq()
    qc_ion = transpile(qc, basis_gates=ionq_gates, optimization_level=opt_level)
    score_ionq = calc_score_from_str(qc_ion.qasm(), ionq)

    qc_rigetti = transpile(qc, basis_gates=rigetti_gates, optimization_level=opt_level, coupling_map=get_cmap_rigetti_m1(10))
    rigetti_m1 = get_rigetti_m1()
    score_rigetti = calc_score_from_str(qc_rigetti.qasm(), rigetti_m1)

    #print("Scores: ", [score_ibm, score_ionq, score_rigetti])


    return [score_ibm, score_ionq, score_rigetti]