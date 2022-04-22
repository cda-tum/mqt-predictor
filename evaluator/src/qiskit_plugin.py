from qiskit import transpile
from evaluator.src.utils import *
from qiskit.test.mock import FakeMontreal, FakeWashington


def get_qiskit_gates(qc):
    opt_level = 2
    gates_ibm_washington = get_ibm_washington_gates(qc, opt_level)
    gates_ibm_montreal = get_ibm_montreal_gates(qc, opt_level)
    gates_ionq = get_ionq_gates(qc, opt_level)
    gates_rigetti = get_rigetti_gates(qc, opt_level)
    gates_oqc = get_oqc_gates(qc, opt_level)

    return (
        "qiskit",
        [
            (gates_ibm_washington, "ibm_washington"),
            (gates_ibm_montreal, "ibm_montreal"),
            (gates_ionq, "ionq"),
            (gates_rigetti, "rigetti_m1"),
            (gates_oqc, "oqc_lucy"),
        ],
    )


def get_ibm_washington_gates(qc, opt_level):
    ibm_washington = get_ibm_washington()
    if qc.num_qubits > ibm_washington["num_qubits"]:
        gates_ibm_washington = None
    else:
        qc_ibm = transpile(
            qc,
            basis_gates=get_ibm_native_gates(),
            optimization_level=opt_level,
            coupling_map=FakeWashington().configuration().coupling_map,
            seed_transpiler=10,
            layout_method="sabre",
            routing_method="sabre",
        )
        gates_ibm_washington = count_qubit_gates_ibm(qc_ibm, "ibm")

    return gates_ibm_washington


def get_ibm_montreal_gates(qc, opt_level):
    ibm_montreal = get_ibm_montreal()
    if qc.num_qubits > ibm_montreal["num_qubits"]:
        gates_ibm_montreal = None
    else:
        qc_ibm = transpile(
            qc,
            basis_gates=get_ibm_native_gates(),
            optimization_level=opt_level,
            coupling_map=FakeMontreal().configuration().coupling_map,
            seed_transpiler=10,
            layout_method="sabre",
            routing_method="sabre",
        )
        gates_ibm_montreal = count_qubit_gates_ibm(qc_ibm, "ibm")

    return gates_ibm_montreal


def get_ionq_gates(qc, opt_level):
    ionq = get_ionq()
    if qc.num_qubits > ionq["num_qubits"]:
        gates_ionq = None
    else:
        qc_ion = transpile(
            qc,
            basis_gates=get_ionq_native_gates(),
            optimization_level=opt_level,
            seed_transpiler=10,
            layout_method="sabre",
            routing_method="sabre",
        )
        gates_ionq = count_qubit_gates_ibm(qc_ion, "ionq")

    return gates_ionq


def get_rigetti_gates(qc, opt_level):
    rigetti_m1 = get_rigetti_m1()
    if qc.num_qubits > rigetti_m1["num_qubits"]:
        gates_rigetti = None
    else:
        qc_rigetti = transpile(
            qc,
            basis_gates=get_rigetti_native_gates(),
            optimization_level=opt_level,
            coupling_map=get_cmap_rigetti_m1(10),
            seed_transpiler=10,
            layout_method="sabre",
            routing_method="sabre",
        )
        gates_rigetti = count_qubit_gates_ibm(qc_rigetti, "rigetti")
    return gates_rigetti


def get_oqc_gates(qc, opt_level):
    oqc_lucy = get_oqc_lucy()
    if qc.num_qubits > oqc_lucy["num_qubits"]:
        gates_oqc = None
    else:
        qc_oqc = transpile(
            qc,
            basis_gates=get_oqc_native_gates(),
            optimization_level=opt_level,
            coupling_map=get_c_map_oqc_lucy(),
            seed_transpiler=10,
            layout_method="sabre",
            routing_method="sabre",
        )
        gates_oqc = count_qubit_gates_ibm(qc_oqc, "oqc")

    return gates_oqc


def get_ibm_native_gates():
    ibm_gates = ["rz", "sx", "x", "cx"]
    return ibm_gates


def get_rigetti_native_gates():
    rigetti_gates = ["rx", "rz", "cz"]
    return rigetti_gates


def get_ionq_native_gates():
    ionq_gates = ["rxx", "rz", "ry", "rx"]
    return ionq_gates


def get_oqc_native_gates():
    oqc_gates = ["rz", "sx", "x", "ecr"]
    return oqc_gates
