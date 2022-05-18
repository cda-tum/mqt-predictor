from predictor.src import utils
from predictor.src import pytket_plugin, qiskit_plugin

import pytest

from mqt.bench import benchmark_generator
from pytket.extensions.qiskit import qiskit_to_tk


def test_get_machines():
    assert len(utils.get_machines()) == 19


def test_get_openqasm_gates():
    assert len(utils.get_openqasm_gates()) == 42


def test_get_width_penalty():
    assert utils.get_width_penalty() >= 0


def test_get_width_penalty():
    assert utils.get_width_penalty() > 0


def test_get_cmaps():
    assert utils.get_cmap_oqc_lucy() == [
        [0, 1],
        [0, 7],
        [1, 2],
        [2, 3],
        [7, 6],
        [6, 5],
        [4, 3],
        [4, 5],
    ]
    assert not utils.get_cmap_rigetti_m1() is None


def test_get_openqasm_gates():
    assert utils.get_openqasm_gates() == [
        "u3",
        "u2",
        "u1",
        "cx",
        "id",
        "u0",
        "u",
        "p",
        "x",
        "y",
        "z",
        "h",
        "s",
        "sdg",
        "t",
        "tdg",
        "rx",
        "ry",
        "rz",
        "sx",
        "sxdg",
        "cz",
        "cy",
        "swap",
        "ch",
        "ccx",
        "cswap",
        "crx",
        "cry",
        "crz",
        "cu1",
        "cp",
        "cu3",
        "csx",
        "cu",
        "rxx",
        "rzz",
        "rccx",
        "rc3x",
        "c3x",
        "c3sqrtx",
        "c4x",
    ]


def test_get_machines():
    assert utils.get_machines() == [
        "qiskit_ionq_opt2",
        "qiskit_ibm_washington_opt2",
        "qiskit_ibm_montreal_opt2",
        "qiskit_rigetti_opt2",
        "qiskit_oqc_opt2",
        "qiskit_ionq_opt3",
        "qiskit_ibm_washington_opt3",
        "qiskit_ibm_montreal_opt3",
        "qiskit_rigetti_opt3",
        "qiskit_oqc_opt3",
        "tket_ionq",
        "tket_ibm_washington_line",
        "tket_ibm_montreal_line",
        "tket_rigetti_line",
        "tket_oqc_line",
        "tket_ibm_washington_graph",
        "tket_ibm_montreal_graph",
        "tket_rigetti_graph",
        "tket_oqc_graph",
    ]
    assert len(utils.get_machines()) == 19


def test_rigetti_fids():
    fid1 = utils.get_rigetti_m1_fid1()
    fid2 = utils.get_rigetti_m1_fid2()
    assert fid1 > 0 and fid1 < 1
    assert fid2 > 0 and fid2 < 1


def test_qubit_counts():

    qc = benchmark_generator.get_one_benchmark("dj", 1, 5)
    num_qubits = qc.num_qubits

    qiskit_gates = qiskit_plugin.save_qiskit_compiled_circuits(
        qc, 2, 10, "dj_indep_5.qasm"
    )
    assert qiskit_gates
    qc_tket = qiskit_to_tk(qc)
    tket_gates = pytket_plugin.save_tket_compiled_circuits(
        qc_tket, True, 10, "dj_indep_5.qasm"
    )
    assert tket_gates
