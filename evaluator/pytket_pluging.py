from pytket.extensions.qiskit import IBMQBackend
from pytket.extensions.aqt import AQTBackend
from pytket.extensions.qiskit import IBMQBackend
from pytket.extensions.pyquil import ForestBackend, ForestStateBackend
from pytket.extensions.ionq import IonQBackend
from pytket.circuit.display import render_circuit_jupyter
from pytket.passes import PlacementPass, RoutingPass, SequencePass, FullPeepholeOptimise, SynthesiseTket
from pytket.placement import LinePlacement
from pytket.placement import GraphPlacement
import pytket
from pytket import qasm
from qiskit import QuantumCircuit
from utils import calc_score_from_str, calc_score, get_rigetti_c_map, get_cmap_imbq_washington, get_ibm_washington, get_rigetti_m1, get_ionq


def get_tket_scores(qc_filepath: QuantumCircuit, opt_level=0):
    ibm_arch = pytket.architecture.Architecture(get_cmap_imbq_washington())
    rigetti_arch = pytket.architecture.Architecture(get_rigetti_c_map(10))

    b1 = IBMQBackend("ibmq_santiago")
    b2 = AQTBackend(device_name="sim/noise-model-1", access_token='')
    b3 = IonQBackend(api_key="")
    b4 = ForestStateBackend()
    c = qasm.circuit_from_qasm(qc_filepath)

    b1.rebase_pass().apply(c)
    PlacementPass(LinePlacement(ibm_arch)).apply(c)
    RoutingPass(ibm_arch).apply(c)
    if opt_level == 1:
        SynthesiseTket().apply(c)
    elif opt_level == 2:
        FullPeepholeOptimise().apply(c)
    b1.rebase_pass().apply(c)
    qc_qasm = qasm.circuit_to_qasm_str(c)
    ibm_washington = get_ibm_washington()
    score_ibm = calc_score_from_str(qc_qasm, ibm_washington)

    c = qasm.circuit_from_qasm(qc_filepath)
    b2.rebase_pass().apply(c)

    qc_qasm = qasm.circuit_to_qasm_str(c)
    ionq=get_ionq()
    score_aqt = calc_score_from_str(qc_qasm, ionq)

    c = qasm.circuit_from_qasm(qc_filepath)
    b3.rebase_pass().apply(c)
    qc_qasm = qasm.circuit_to_qasm_str(c)
    score_ionq = calc_score_from_str(qc_qasm, ionq)

    c = qasm.circuit_from_qasm(qc_filepath)
    b4.rebase_pass().apply(c)
    PlacementPass(LinePlacement(rigetti_arch)).apply(c)
    RoutingPass(ibm_arch).apply(c)
    b4.rebase_pass().apply(c)
    qc_qasm = qasm.circuit_to_qasm_str(c)
    rigetti_m1 = get_rigetti_m1()
    score_rigetti = calc_score_from_str(qc_qasm, rigetti_m1)

    return [score_ibm, score_aqt, score_ionq, score_rigetti]