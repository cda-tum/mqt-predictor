from predictor.src import utils
from predictor.src import pytket_plugin, qiskit_plugin

import pytest

from mqt.bench import benchmark_generator
from pytket.extensions.qiskit import qiskit_to_tk


def test_get_machines():
    assert len(utils.get_machines()) == 10


def test_get_openqasm_gates():
    assert len(utils.get_openqasm_gates()) == 42


@pytest.mark.parametrize(
    "backend", ["ibm_washington", "ibm_montreal", "ionq", "rigetti_m1", "oqc_lucy"]
)
def test_get_backend_information(backend: str):
    assert not utils.get_backend_information(backend) is None


def test_get_width_penalty():
    assert utils.get_width_penalty() >= 0


def test_get_width_penalty():
    assert utils.get_width_penalty() > 0


def test_get_backend_information():
    assert not utils.get_backend_information("ibm_washington") is None
    assert not utils.get_backend_information("ibm_montreal") is None
    assert not utils.get_backend_information("ionq") is None
    assert not utils.get_backend_information("rigetti_m1") is None
    assert not utils.get_backend_information("oqc_lucy") is None


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
    assert not utils.get_cmap_rigetti_m1(10) is None


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
        "qiskit_ibm_washington",
        "qiskit_ibm_montreal",
        "qiskit_ionq",
        "qiskit_rigetti",
        "qiskit_oqc",
        "tket_ibm_washington",
        "tket_ibm_montreal",
        "tket_ionq",
        "tket_rigetti",
        "tket_oqc",
    ]


def test_rigetti_fids():
    fid1 = utils.get_rigetti_m1_fid1()
    fid2 = utils.get_rigetti_m1_fid2()
    assert fid1 > 0 and fid1 < 1
    assert fid2 > 0 and fid2 < 1


def test_qubit_counts():

    qc = benchmark_generator.get_one_benchmark("dj", 1, 5)
    num_qubits = qc.num_qubits

    qiskit_gates = qiskit_plugin.get_qiskit_gates(qc)
    assert not qiskit_gates is None
    qc_tket = qiskit_to_tk(qc)
    tket_gates = pytket_plugin.get_tket_gates(qc_tket)
    assert not tket_gates is None
