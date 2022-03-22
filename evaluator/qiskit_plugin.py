from qiskit import transpile, QuantumCircuit
from utils import calc_score_from_str, calc_score, get_rigetti_c_map, get_cmap_imbq_washington, get_ibm_washington, get_rigetti_m1, get_ionq

def get_qiskit_scores(qc_filepath:QuantumCircuit, opt_level=0):

    ibm_gates = ['id', 'rz', 'sx', 'x', 'cx', 'reset']
    rigetti_gates = ["rx", "rz", "cz"]
    ionq_gates = ["rxx", "rz", "ry", "rx"]
    aqt_gates = ['rx', 'ry', 'rxx']

    qc = QuantumCircuit.from_qasm_file(qc_filepath)
    #compile to ibm architecture
    qc_ibm = transpile(qc, basis_gates=ibm_gates, optimization_level=opt_level)

    ibm_washington = get_ibm_washington()
    score_ibm = calc_score_from_str(qc_ibm.qasm(), ibm_washington)
    #compile to rigetti architecture
    qc_rigetti = transpile(qc, basis_gates=rigetti_gates, optimization_level=opt_level)

    rigetti_m1 = get_rigetti_m1()
    score_rigetti = calc_score_from_str(qc_rigetti.qasm(), rigetti_m1)
    #compile to aqt architecture
    qc_aqt = transpile(qc, basis_gates=aqt_gates, optimization_level=opt_level)
    ionq = get_ionq()
    score_aqt = calc_score_from_str(qc_aqt.qasm(), ionq)
    #compile to rigetti architecture
    qc_ion = transpile(qc, basis_gates=ionq_gates, optimization_level=opt_level)

    score_ionq = calc_score_from_str(qc_ion.qasm(), ionq)

    return [score_ibm, score_aqt, score_ionq, score_rigetti]