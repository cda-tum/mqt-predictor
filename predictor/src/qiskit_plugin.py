from qiskit import transpile
from predictor.src.utils import *
from qiskit.test.mock import FakeMontreal, FakeWashington


def save_qiskit_compiled_circuits(
    qc, opt_level: int, timeout: int, benchmark_name: str
):
    offset = 0
    if opt_level == 3:
        offset = 5
    try:
        path = get_compiled_output_folder()

        ionq = timeout_watcher(get_ionq_qc, [qc, opt_level], timeout)
        if ionq:
            filename = (
                path
                + benchmark_name.split(".")[0]
                + "_ionq"
                + "_qiskit_opt_"
                + str(opt_level)
                + "_"
                + str(0 + offset)
                + ".qasm"
            )
            ionq.qasm(filename=filename)

        ibm_washington = timeout_watcher(
            get_ibm_washington_qc, [qc, opt_level], timeout
        )
        if ibm_washington:
            filename = (
                path
                + benchmark_name.split(".")[0]
                + "_ibm_washington"
                + "_qiskit_opt_"
                + str(opt_level)
                + "_"
                + str(1 + offset)
                + ".qasm"
            )
            ibm_washington.qasm(filename=filename)

        ibm_montreal = timeout_watcher(get_ibm_montreal_qc, [qc, opt_level], timeout)
        if ibm_montreal:
            filename = (
                path
                + benchmark_name.split(".")[0]
                + "_ibm_montreal"
                + "_qiskit_opt_"
                + str(opt_level)
                + "_"
                + str(2 + offset)
                + ".qasm"
            )
            ibm_montreal.qasm(filename=filename)

        rigetti = timeout_watcher(get_rigetti_qc, [qc, opt_level], timeout)
        if rigetti:
            filename = (
                path
                + benchmark_name.split(".")[0]
                + "_rigetti"
                + "_qiskit_opt_"
                + str(opt_level)
                + "_"
                + str(3 + offset)
                + ".qasm"
            )
            rigetti.qasm(filename=filename)

        oqc = timeout_watcher(get_oqc_qc, [qc, opt_level], timeout)
        if oqc:
            filename = (
                path
                + benchmark_name.split(".")[0]
                + "_oqc"
                + "_qiskit_opt_"
                + str(opt_level)
                + "_"
                + str(4 + offset)
                + ".qasm"
            )
            oqc.qasm(filename=filename)
    except:
        return False
    else:
        return [ibm_washington, ibm_montreal, rigetti, oqc, ionq]


def get_ibm_washington_qc(qc, opt_level):
    ibm_washington = get_ibm_washington()
    if qc.num_qubits > ibm_washington["num_qubits"]:
        return None
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

    return qc_ibm


def get_ibm_montreal_qc(qc, opt_level):
    ibm_montreal = get_ibm_montreal()
    if qc.num_qubits > ibm_montreal["num_qubits"]:
        return None
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

    return qc_ibm


def get_ionq_qc(qc, opt_level):
    ionq = get_ionq()
    if qc.num_qubits > ionq["num_qubits"]:
        return None
    else:
        qc_ion = transpile(
            qc,
            basis_gates=get_ionq_native_gates(),
            optimization_level=opt_level,
            seed_transpiler=10,
            layout_method="sabre",
            routing_method="sabre",
        )

    return qc_ion


def get_rigetti_qc(qc, opt_level):
    rigetti_m1 = get_rigetti_m1()
    if qc.num_qubits > rigetti_m1["num_qubits"]:
        return None
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

    return qc_rigetti


def get_oqc_qc(qc, opt_level):

    oqc_lucy = get_oqc_lucy()
    if qc.num_qubits > oqc_lucy["num_qubits"]:
        return None
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
    return qc_oqc


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
