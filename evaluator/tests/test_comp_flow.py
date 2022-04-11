from evaluator import evaluator
from evaluator.src import utils, pytket_plugin, qiskit_plugin
import pytest
from pytket.extensions.qiskit import qiskit_to_tk
from mqt.bench import get_one_benchmark


def test_qiskit_native_gatesets():
    assert qiskit_plugin.get_ibm_native_gates() == ["rz", "sx", "x", "cx"]
    assert qiskit_plugin.get_rigetti_native_gates() == ["rx", "rz", "cz"]
    assert qiskit_plugin.get_ionq_native_gates() == ["rxx", "rz", "ry", "rx"]
    assert qiskit_plugin.get_oqc_native_gates() == ["rz", "sx", "x", "ecr"]


def test_qiskit_gate_counts():
    qc = get_one_benchmark("dj", 1, 3)
    get_ibm_montreal_gates = qiskit_plugin.get_ibm_montreal_gates(qc, 0)
    assert not get_ibm_montreal_gates is None
    ibm_washington_gates = qiskit_plugin.get_ibm_washington_gates(qc, 0)
    assert not ibm_washington_gates is None
    get_ionq_gates = qiskit_plugin.get_ionq_gates(qc, 0)
    assert not get_ionq_gates is None
    get_rigetti_gates = qiskit_plugin.get_rigetti_gates(qc, 0)
    assert not get_rigetti_gates is None
    get_oqc_gates = qiskit_plugin.get_oqc_gates(qc, 0)
    assert not get_oqc_gates is None

    qc = get_one_benchmark("dj", 1, 13)
    get_ibm_montreal_gates = qiskit_plugin.get_ibm_montreal_gates(qc, 0)
    assert not get_ibm_montreal_gates is None
    ibm_washington_gates = qiskit_plugin.get_ibm_washington_gates(qc, 0)
    assert not ibm_washington_gates is None
    get_ionq_gates = qiskit_plugin.get_ionq_gates(qc, 0)
    assert get_ionq_gates is None
    get_rigetti_gates = qiskit_plugin.get_rigetti_gates(qc, 0)
    assert not get_rigetti_gates is None
    get_oqc_gates = qiskit_plugin.get_oqc_gates(qc, 0)
    assert get_oqc_gates is None


def test_tket_gate_counts():
    qc = get_one_benchmark("dj", 1, 3)
    qc_tket = qiskit_to_tk(qc)
    get_ibm_montreal_gates = pytket_plugin.get_ibm_montreal_gates(qc_tket, 0)
    assert not get_ibm_montreal_gates is None
    ibm_washington_gates = pytket_plugin.get_ibm_washington_gates(qc_tket, 0)
    assert not ibm_washington_gates is None
    get_ionq_gates = pytket_plugin.get_ionq_gates(qc_tket, 0)
    assert not get_ionq_gates is None
    get_rigetti_gates = pytket_plugin.get_rigetti_gates(qc_tket, 0)
    assert not get_rigetti_gates is None
    get_oqc_gates = pytket_plugin.get_oqc_gates(qc_tket, 0)
    assert not get_oqc_gates is None

    qc = get_one_benchmark("dj", 1, 13)
    qc_tket = qiskit_to_tk(qc)
    get_ibm_montreal_gates = pytket_plugin.get_ibm_montreal_gates(qc_tket, 0)
    assert not get_ibm_montreal_gates is None
    ibm_washington_gates = pytket_plugin.get_ibm_washington_gates(qc_tket, 0)
    assert not ibm_washington_gates is None
    get_ionq_gates = pytket_plugin.get_ionq_gates(qc_tket, 0)
    assert get_ionq_gates is None
    get_rigetti_gates = pytket_plugin.get_rigetti_gates(qc_tket, 0)
    assert not get_rigetti_gates is None
    get_oqc_gates = pytket_plugin.get_oqc_gates(qc_tket, 0)
    assert get_oqc_gates is None


def test_get_qiskit_gates():
    qc = get_one_benchmark("ghz", 1, 5)
    res = qiskit_plugin.get_qiskit_gates(qc)
    assert res[0] == "qiskit"
    assert len(res[1]) == 5


def test_get_tket_gates():
    qc = get_one_benchmark("ghz", 1, 5)
    qc_tket = qiskit_to_tk(qc)
    res = pytket_plugin.get_tket_gates(qc_tket)
    assert res[0] == "tket"
    assert len(res[1]) == 5
