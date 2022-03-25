import numpy as np
from pytket.extensions.aqt import AQTBackend
from pytket.extensions.qiskit import IBMQBackend
from pytket.extensions.pyquil import ForestStateBackend
from pytket.extensions.ionq import IonQBackend
from pytket.passes import PlacementPass, RoutingPass, FullPeepholeOptimise, SynthesiseTket
from pytket.placement import LinePlacement

from pytket import qasm
from evaluator.utils import calc_score_from_str, calc_score, get_cmap_rigetti_m1, get_cmap_imbq_washington, get_ibm_washington, get_rigetti_m1, get_ionq
import pytket
from pytket import OpType, _tket, Circuit
from pytket.passes import RebaseCustom


def get_tket_scores(qasm_qc, opt_level=0):
    qc = qasm.circuit_from_qasm_str(qasm_qc)
    score_ibm = get_ibm_score(qc, opt_level)

    qc = qasm.circuit_from_qasm_str(qasm_qc)
    score_ionq = get_ionq_score(qc)

    qc = qasm.circuit_from_qasm_str(qasm_qc)
    score_rigetti = get_rigetti_score(qc, opt_level)

    #print("Scores: ", [score_ibm, score_ionq, score_rigetti])

    return [score_ibm, score_ionq, score_rigetti]


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
    qc_qasm = qasm.circuit_to_qasm_str(qc)
    rigetti_m1 = get_rigetti_m1()
    score_rigetti = calc_score_from_str(qc_qasm, rigetti_m1)
    return score_rigetti


def get_ionq_score(qc):
    #b3 = IonQBackend(api_key="")
    #b3.rebase_pass().apply(qasm_qc)
    ionq_rebase = get_ionq_rebase()
    ionq_rebase.apply(qc)
    qc_qasm = qasm.circuit_to_qasm_str(qc)
    ionq = get_ionq()
    score_ionq = calc_score_from_str(qc_qasm, ionq)
    return score_ionq


def get_aqt_score(qc):
    backend = AQTBackend(device_name="sim/noise-model-1", access_token='')
    backend.rebase_pass().apply(qc)
    qc_qasm = qasm.circuit_to_qasm_str(qc)
    ionq = get_ionq()
    score_aqt = calc_score_from_str(qc_qasm, ionq)
    return score_aqt


def get_ibm_score(qc, opt_level):
    ibm_arch = pytket.architecture.Architecture(get_cmap_imbq_washington())
    backend = IBMQBackend("ibmq_santiago")
    backend.rebase_pass().apply(qc)
    PlacementPass(LinePlacement(ibm_arch)).apply(qc)
    RoutingPass(ibm_arch).apply(qc)
    if opt_level == 1:
        SynthesiseTket().apply(qc)
    elif opt_level == 2:
        FullPeepholeOptimise().apply(qc)
    backend.rebase_pass().apply(qc)
    qc_qasm = qasm.circuit_to_qasm_str(qc)
    ibm_washington = get_ibm_washington()
    score_ibm = calc_score_from_str(qc_qasm, ibm_washington)
    return score_ibm


def get_ionq_rebase():
    ionq_gateset = {OpType.Rz, OpType.Ry, OpType.XXPhase}
    cx_in_ionq = _tket.circuit._library._CX_using_XXPhase_0()

    def tk1_to_rzry(a, b, c):
        circ = Circuit(1)
        circ.Rz(c + 0.5, 0).Ry(b, 0).Rz(a - 0.5, 0)
        return circ

    ionq_rebase = RebaseCustom(ionq_gateset, cx_in_ionq, tk1_to_rzry)
    return ionq_rebase