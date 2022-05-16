from qiskit import transpile
from predictor.src.utils import *
from qiskit.test.mock import FakeMontreal, FakeWashington


def get_qiskit_gates(qc, opt_level: int, timeout: int):
    opt_level = opt_level

    gates_ibm_washington = timeout_watcher(
        get_ibm_washington_gates, [qc, opt_level], timeout
    )

    gates_ibm_montreal = timeout_watcher(
        get_ibm_montreal_gates, [qc, opt_level], timeout
    )

    gates_ionq = timeout_watcher(get_ionq_gates, [qc, opt_level], timeout)

    gates_rigetti = timeout_watcher(get_rigetti_gates, [qc, opt_level], timeout)

    gates_oqc = timeout_watcher(get_oqc_gates, [qc, opt_level], timeout)

    return (
        "qiskit_opt" + str(opt_level),
        [
            (gates_ionq, "ionq"),
            (gates_ibm_washington, "ibm_washington"),
            (gates_ibm_montreal, "ibm_montreal"),
            (gates_rigetti, "rigetti_m1"),
            (gates_oqc, "oqc_lucy"),
        ],
    )


def get_ibm_washington_gates(qc, opt_level, return_circuit: bool = False):
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
        gates_ibm_washington = count_qubit_gates_qiskit(qc_ibm, "ibm")

    if return_circuit:
        return qc_ibm.qasm()
    return gates_ibm_washington


def get_ibm_montreal_gates(qc, opt_level, return_circuit: bool = False):
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
        gates_ibm_montreal = count_qubit_gates_qiskit(qc_ibm, "ibm")

    if return_circuit:
        return qc_ibm.qasm()
    return gates_ibm_montreal


def get_ionq_gates(qc, opt_level, return_circuit: bool = False):
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
        gates_ionq = count_qubit_gates_qiskit(qc_ion, "ionq")

    if return_circuit:
        return qc_ion.qasm()
    return gates_ionq


def get_rigetti_gates(qc, opt_level, return_circuit: bool = False):
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
        gates_rigetti = count_qubit_gates_qiskit(qc_rigetti, "rigetti")

    if return_circuit:
        return qc_rigetti.qasm()
    return gates_rigetti


def get_oqc_gates(qc, opt_level, return_circuit: bool = False):
    oqc_lucy = get_oqc_lucy()
    if qc.num_qubits > oqc_lucy["num_qubits"]:
        gates_oqc = None
    else:
        qc_oqc = transpile(
            qc,
            basis_gates=get_oqc_native_gates(),
            optimization_level=opt_level,
            coupling_map=get_cmap_oqc_lucy(),
            seed_transpiler=10,
            layout_method="sabre",
            routing_method="sabre",
        )
        gates_oqc = count_qubit_gates_qiskit(qc_oqc, "oqc")

    if return_circuit:
        return qc_oqc.qasm()
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
