from pytket.passes import (
    PlacementPass,
    RoutingPass,
    FullPeepholeOptimise,
    SynthesiseTket,
)
from pytket.placement import LinePlacement
from pytket import architecture
from pytket.passes import auto_rebase_pass

from qiskit.test.mock import FakeMontreal
from evaluator.src.utils import *


def get_tket_gates(qc, opt_level=0):

    gates_ibm_washington = get_ibm_washington_gates(qc, opt_level)
    gates_ibm_montreal = get_ibm_montreal_gates(qc, opt_level)
    gates_ionq = get_ionq_gates(qc, opt_level)
    gates_rigetti = get_rigetti_gates(qc, opt_level)
    gates_oqc = get_oqc_gates(qc, opt_level)

    return (
        "tket",
        [
            (gates_ibm_washington, "ibm_washington"),
            (gates_ibm_montreal, "ibm_montreal"),
            (gates_ionq, "ionq"),
            (gates_rigetti, "rigetti_m1"),
            (gates_oqc, "oqc_lucy"),
        ],
    )


def get_rigetti_gates(qc, opt_level):
    if qc.n_qubits > get_rigetti_m1()["num_qubits"]:
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
        assert sum(gates_rigetti) == qc.n_gates - qc.n_gates_of_type(
            OpType.Measure
        ) - qc.n_gates_of_type(OpType.Barrier)
    return gates_rigetti


def get_ionq_gates(qc, opt_level):
    if qc.n_qubits > get_ionq()["num_qubits"]:
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
        assert sum(gates_ionq) == qc.n_gates - qc.n_gates_of_type(
            OpType.Measure
        ) - qc.n_gates_of_type(OpType.Barrier)
    return gates_ionq


def get_oqc_gates(qc, opt_level):
    if qc.n_qubits > get_oqc_lucy()["num_qubits"]:
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
        assert sum(gates_oqc) == qc.n_gates - qc.n_gates_of_type(
            OpType.Measure
        ) - qc.n_gates_of_type(OpType.Barrier)
    return gates_oqc


def get_ibm_washington_gates(qc, opt_level):
    if qc.n_qubits > get_ibm_washington()["num_qubits"]:
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
        assert sum(gates_ibm_washington) == qc.n_gates - qc.n_gates_of_type(
            OpType.Measure
        ) - qc.n_gates_of_type(OpType.Barrier)
    return gates_ibm_washington


def get_ibm_montreal_gates(qc, opt_level):
    if qc.n_qubits > get_ibm_montreal()["num_qubits"]:
        gates_ibm_montreal = None
    else:
        ibm_montreal_arch = architecture.Architecture(
            FakeMontreal().configuration().coupling_map
        )
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
        assert sum(gates_ibm_montreal) == qc.n_gates - qc.n_gates_of_type(
            OpType.Measure
        ) - qc.n_gates_of_type(OpType.Barrier)
    return gates_ibm_montreal


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
