from pytket.extensions.aqt import AQTBackend
from pytket.passes import PlacementPass, RoutingPass, FullPeepholeOptimise, SynthesiseTket
from pytket.placement import LinePlacement
from pytket import _tket, Circuit, architecture, qasm
from pytket.passes import auto_rebase_pass

from qiskit.test.mock import FakeMontreal
from evaluator.utils import *


def get_tket_scores(qc, opt_level=0):
    penalty_width = 1000000

    if qc.n_qubits > get_ibm_washington()['num_qubits']:
        score_ibm_washington = penalty_width
    else:
        score_ibm_washington = get_ibm_washington_score(qc, opt_level)

    if qc.n_qubits > get_ibm_montreal()['num_qubits']:
        score_ibm_montreal = penalty_width
    else:
        score_ibm_montreal = get_ibm_montreal_score(qc, opt_level)

    if qc.n_qubits > get_ionq()['num_qubits']:
        score_ionq = penalty_width
    else:
        score_ionq = get_ionq_score(qc, opt_level)

    if qc.n_qubits > get_rigetti_m1()['num_qubits']:
        score_rigetti = penalty_width
    else:
        score_rigetti = get_rigetti_score(qc, opt_level)

    if qc.n_qubits > get_oqc_lucy()['num_qubits']:
        score_oqc = penalty_width
    else:
        score_oqc = get_oqc_score(qc, opt_level)

    print("Scores TKET: ", [score_ibm_washington, score_ibm_montreal, score_ionq, score_rigetti, score_oqc])

    return [score_ibm_washington, score_ibm_montreal, score_ionq, score_rigetti, score_oqc]

def get_tket_gates(qc, opt_level=0):

    gates_ibm_washington = get_ibm_washington_gates(qc, opt_level)
    gates_ibm_montreal = get_ibm_montreal_gates(qc, opt_level)
    gates_ionq = get_ionq_gates(qc, opt_level)
    gates_rigetti = get_rigetti_gates(qc, opt_level)
    gates_oqc = get_oqc_gates(qc, opt_level)


    return ("tket", [gates_ibm_washington, gates_ibm_montreal, gates_ionq, gates_rigetti,  gates_oqc])

def get_rigetti_score(qc, opt_level):
    backend = get_rigetti_rebase()
    rigetti_arch = architecture.Architecture(get_cmap_rigetti_m1(10))
    backend.apply(qc)
    PlacementPass(LinePlacement(rigetti_arch)).apply(qc)
    RoutingPass(rigetti_arch).apply(qc)
    if opt_level == 1:
        SynthesiseTket().apply(qc)
    elif opt_level == 2:
        FullPeepholeOptimise().apply(qc)
    backend.apply(qc)
    rigetti_m1 = get_rigetti_m1()
    score_rigetti = calc_score_from_qc(qc, rigetti_m1, "tket")
    return score_rigetti

def get_rigetti_gates(qc, opt_level):
    if qc.n_qubits > get_rigetti_m1()['num_qubits']:
        gates_rigetti = None
    else:
        backend = get_rigetti_rebase()
        rigetti_arch = architecture.Architecture(get_cmap_rigetti_m1(10))
        backend.apply(qc)
        PlacementPass(LinePlacement(rigetti_arch)).apply(qc)
        RoutingPass(rigetti_arch).apply(qc)
        if opt_level == 1:
            SynthesiseTket().apply(qc)
        elif opt_level == 2:
            FullPeepholeOptimise().apply(qc)
        backend.apply(qc)
        gates_rigetti = count_qubit_gates_tket(qc, "rigetti")
    return (gates_rigetti, "rigetti_m1")


def get_ionq_score(qc, opt_level):
    ionq_rebase = get_ionq_rebase()
    ionq_rebase.apply(qc)
    if opt_level == 1:
        SynthesiseTket().apply(qc)
        ionq_rebase.apply(qc)
    elif opt_level == 2:
        FullPeepholeOptimise().apply(qc)
        ionq_rebase.apply(qc)
    ionq = get_ionq()
    score_ionq = calc_score_from_qc(qc, ionq, "tket")
    return score_ionq

def get_ionq_gates(qc, opt_level):
    if qc.n_qubits > get_ionq()['num_qubits']:
        gates_ionq = None
    else:
        ionq_rebase = get_ionq_rebase()
        ionq_rebase.apply(qc)
        if opt_level == 1:
            SynthesiseTket().apply(qc)
            ionq_rebase.apply(qc)
        elif opt_level == 2:
            FullPeepholeOptimise().apply(qc)
            ionq_rebase.apply(qc)
        gates_ionq = count_qubit_gates_tket(qc, "ionq")
    return (gates_ionq, "ionq")

def get_oqc_score(qc, opt_level):
    oqc_rebase = get_oqc_rebase()
    oqc_rebase.apply(qc)
    if opt_level == 1:
        SynthesiseTket().apply(qc)
        oqc_rebase.apply(qc)
    elif opt_level == 2:
        FullPeepholeOptimise().apply(qc)
        oqc_rebase.apply(qc)
    oqc_lucy = get_oqc_lucy()
    score_ionq = calc_score_from_qc(qc, oqc_lucy, "tket")
    return score_ionq

def get_oqc_gates(qc, opt_level):
    if qc.n_qubits > get_oqc_lucy()['num_qubits']:
        gates_oqc = None
    else:
        oqc_rebase = get_oqc_rebase()
        oqc_rebase.apply(qc)
        if opt_level == 1:
            SynthesiseTket().apply(qc)
            oqc_rebase.apply(qc)
        elif opt_level == 2:
            FullPeepholeOptimise().apply(qc)
            oqc_rebase.apply(qc)
        gates_oqc = count_qubit_gates_tket(qc, "oqc")
    return (gates_oqc, "oqc_lucy")


def get_aqt_score(qc, opt_level):
    backend = AQTBackend(device_name="sim/noise-model-1", access_token='')
    backend.rebase_pass().apply(qc)
    if opt_level == 1:
        SynthesiseTket().apply(qc)
        backend.rebase_pass().apply(qc)
    elif opt_level == 2:
        FullPeepholeOptimise().apply(qc)
        backend.apply(qc)
    ionq = get_ionq()
    gates_aqt = count_qubit_gates_tket(qc, ionq)
    return (gates_aqt, "ionq")


def get_ibm_washington_score(qc, opt_level):
    ibm_washington_arch = architecture.Architecture(get_cmap_imbq_washington())
    backend = get_ibm_rebase()
    backend.apply(qc)
    PlacementPass(LinePlacement(ibm_washington_arch)).apply(qc)
    RoutingPass(ibm_washington_arch).apply(qc)
    if opt_level == 1:
        SynthesiseTket().apply(qc)
    elif opt_level == 2:
        FullPeepholeOptimise().apply(qc)
    backend.apply(qc)
    ibm_washington = get_ibm_washington()
    score_ibm_washington = calc_score_from_qc(qc, ibm_washington, "tket")
    return score_ibm_washington

def get_ibm_washington_gates(qc, opt_level):
    if qc.n_qubits > get_ibm_washington()['num_qubits']:
        gates_ibm_washington = None
    else:
        ibm_washington_arch = architecture.Architecture(get_cmap_imbq_washington())
        backend = get_ibm_rebase()
        backend.apply(qc)
        PlacementPass(LinePlacement(ibm_washington_arch)).apply(qc)
        RoutingPass(ibm_washington_arch).apply(qc)
        if opt_level == 1:
            SynthesiseTket().apply(qc)
        elif opt_level == 2:
            FullPeepholeOptimise().apply(qc)
        backend.apply(qc)
        gates_ibm_washington = count_qubit_gates_tket(qc, "ibm")
    return (gates_ibm_washington, "ibm_washington")

def get_ibm_montreal_score(qc, opt_level):
    ibm_montreal_arch = architecture.Architecture(FakeMontreal().configuration().coupling_map)
    backend = get_ibm_rebase()
    backend.apply(qc)
    PlacementPass(LinePlacement(ibm_montreal_arch)).apply(qc)
    RoutingPass(ibm_montreal_arch).apply(qc)
    if opt_level == 1:
        SynthesiseTket().apply(qc)
    elif opt_level == 2:
        FullPeepholeOptimise().apply(qc)
    backend.apply(qc)
    ibm_montreal = get_ibm_montreal()
    score_ibm_montreal = calc_score_from_qc(qc, ibm_montreal, "tket")
    return score_ibm_montreal

def get_ibm_montreal_gates(qc, opt_level):
    if qc.n_qubits > get_ibm_montreal()['num_qubits']:
        gates_ibm_montreal = None
    else:
        ibm_montreal_arch = architecture.Architecture(FakeMontreal().configuration().coupling_map)
        backend = get_ibm_rebase()
        backend.apply(qc)
        PlacementPass(LinePlacement(ibm_montreal_arch)).apply(qc)
        RoutingPass(ibm_montreal_arch).apply(qc)
        if opt_level == 1:
            SynthesiseTket().apply(qc)
        elif opt_level == 2:
            FullPeepholeOptimise().apply(qc)
        backend.apply(qc)
        ibm_montreal = get_ibm_montreal()
        gates_ibm_montreal = count_qubit_gates_tket(qc, "ibm")
    return (gates_ibm_montreal, "ibm_montreal")


def get_ionq_rebase():
    ionq_gateset = {OpType.Rz, OpType.Ry, OpType.Rx, OpType.XXPhase}
    ionq_rebase = auto_rebase_pass(ionq_gateset)
    return ionq_rebase

def get_oqc_rebase():
    oqc_gateset = {OpType.Rz, OpType.SX, OpType.X, OpType.ECR}
    oqc_rebase = auto_rebase_pass(oqc_gateset)
    return oqc_rebase

def get_rigetti_rebase():
    rigetti_gateset = auto_rebase_pass({OpType.Rz, OpType.Rx, OpType.CZ})
    return rigetti_gateset

def get_ibm_rebase():
    ibm_rebase = auto_rebase_pass({OpType.Rz, OpType.SX, OpType.X, OpType.CX})
    return ibm_rebase