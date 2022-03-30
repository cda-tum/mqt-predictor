import numpy as np
from pytket.extensions.aqt import AQTBackend
from pytket.extensions.qiskit import IBMQBackend
from pytket.extensions.pyquil import ForestStateBackend
from pytket.extensions.ionq import IonQBackend
from pytket.passes import PlacementPass, RoutingPass, FullPeepholeOptimise, SynthesiseTket
from pytket.placement import LinePlacement

from pytket import qasm
from evaluator.utils import *
import pytket
from pytket import OpType, _tket, Circuit
from pytket.passes import RebaseCustom
from pytket.passes._decompositions import _TK1_to_X_SX_Rz
from pytket.extensions.qiskit import tk_to_qiskit

from qiskit.test.mock import FakeMontreal


def get_tket_scores(qasm_qc, opt_level=0):
    penalty_width = 100000

    qc = qasm.circuit_from_qasm_str(qasm_qc)
    if qc.n_qubits > get_ibm_washington()['num_qubits']:
        score_ibm_washington = penalty_width
    else:
        score_ibm_washington = get_ibm_washington_score(qc, opt_level)

    qc = qasm.circuit_from_qasm_str(qasm_qc)
    if qc.n_qubits > get_ibm_montreal()['num_qubits']:
        score_ibm_montreal = penalty_width
    else:
        score_ibm_montreal = get_ibm_montreal_score(qc, opt_level)

    qc = qasm.circuit_from_qasm_str(qasm_qc)
    if qc.n_qubits > get_ionq()['num_qubits']:
        score_ionq = penalty_width
    else:
        score_ionq = get_ionq_score(qc, opt_level)

    qc = qasm.circuit_from_qasm_str(qasm_qc)
    if qc.n_qubits > get_rigetti_m1()['num_qubits']:
        score_rigetti = penalty_width
    else:
        score_rigetti = get_rigetti_score(qc, opt_level)

    qc = qasm.circuit_from_qasm_str(qasm_qc)
    if qc.n_qubits > get_oqc_lucy()['num_qubits']:
        score_oqc = penalty_width
    else:
        score_oqc = get_oqc_score(qc, opt_level)

    print("Scores TKET: ", [score_ibm_washington, score_ibm_montreal, score_ionq, score_rigetti, score_oqc])

    return [score_ibm_washington, score_ibm_montreal, score_ionq, score_rigetti, score_oqc]


def get_rigetti_score(qc, opt_level):
    backend = ForestStateBackend()
    rigetti_arch = pytket.architecture.Architecture(get_cmap_rigetti_m1(10))
    backend.rebase_pass().apply(qc)
    PlacementPass(LinePlacement(rigetti_arch)).apply(qc)
    RoutingPass(rigetti_arch).apply(qc)
    if opt_level == 1:
        SynthesiseTket().apply(qc)
    elif opt_level == 2:
        FullPeepholeOptimise().apply(qc)
    backend.rebase_pass().apply(qc)
    rigetti_m1 = get_rigetti_m1()
    score_rigetti = calc_score(qc, rigetti_m1, "tket")
    return score_rigetti


def get_ionq_score(qc, opt_level):
    #b3 = IonQBackend(api_key="")
    #b3.rebase_pass().apply(qasm_qc)
    ionq_rebase = get_ionq_rebase()
    ionq_rebase.apply(qc)
    if opt_level == 1:
        SynthesiseTket().apply(qc)
        ionq_rebase.apply(qc)
    elif opt_level == 2:
        FullPeepholeOptimise().apply(qc)
        ionq_rebase.apply(qc)
    ionq = get_ionq()
    score_ionq = calc_score(qc, ionq, "tket")
    return score_ionq

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
    score_ionq = calc_score(qc, oqc_lucy, "tket")
    return score_ionq


def get_aqt_score(qc, opt_level):
    backend = AQTBackend(device_name="sim/noise-model-1", access_token='')
    backend.rebase_pass().apply(qc)
    if opt_level == 1:
        SynthesiseTket().apply(qc)
        backend.rebase_pass().apply(qc)
    elif opt_level == 2:
        FullPeepholeOptimise().apply(qc)
        backend.rebase_pass().apply(qc)
    ionq = get_ionq()
    score_aqt = calc_score(qc, ionq)
    return score_aqt


def get_ibm_washington_score(qc, opt_level):
    ibm_washington_arch = pytket.architecture.Architecture(get_cmap_imbq_washington())
    backend = IBMQBackend("ibmq_santiago")
    backend.rebase_pass().apply(qc)
    PlacementPass(LinePlacement(ibm_washington_arch)).apply(qc)
    RoutingPass(ibm_washington_arch).apply(qc)
    if opt_level == 1:
        SynthesiseTket().apply(qc)
    elif opt_level == 2:
        FullPeepholeOptimise().apply(qc)
    backend.rebase_pass().apply(qc)
    ibm_washington = get_ibm_washington()
    score_ibm_washington = calc_score(qc, ibm_washington, "tket")
    return score_ibm_washington

def get_ibm_montreal_score(qc, opt_level):
    ibm_montreal_arch = pytket.architecture.Architecture(FakeMontreal().configuration().coupling_map)
    backend = IBMQBackend("ibmq_santiago")
    backend.rebase_pass().apply(qc)
    PlacementPass(LinePlacement(ibm_montreal_arch)).apply(qc)
    RoutingPass(ibm_montreal_arch).apply(qc)
    if opt_level == 1:
        SynthesiseTket().apply(qc)
    elif opt_level == 2:
        FullPeepholeOptimise().apply(qc)
    backend.rebase_pass().apply(qc)
    ibm_montreal = get_ibm_montreal()
    score_ibm_montreal = calc_score(qc, ibm_montreal, "tket")
    return score_ibm_montreal


def get_ionq_rebase():
    ionq_gateset = {OpType.Rz, OpType.Ry, OpType.XXPhase}
    cx_in_ionq = _tket.circuit._library._CX_using_XXPhase_0()

    def tk1_to_rzry(a, b, c):
        circ = Circuit(1)
        circ.Rz(c + 0.5, 0).Ry(b, 0).Rz(a - 0.5, 0)
        return circ

    ionq_rebase = RebaseCustom(ionq_gateset, cx_in_ionq, tk1_to_rzry)
    return ionq_rebase

def get_oqc_rebase():
    # ["rz", "sx", "x", "ecr"]
    oqc_gateset = {OpType.Rz, OpType.SX, OpType.X, OpType.ECR}
    cx_in_oqc = _tket.circuit._library._CX_using_ECR()
    tk1_to_rz_x_sx = _TK1_to_X_SX_Rz

    oqc_rebase = RebaseCustom(oqc_gateset, cx_in_oqc, tk1_to_rz_x_sx)
    return oqc_rebase