from predictor.src import pytket_plugin, qiskit_plugin
from pytket.extensions.qiskit import qiskit_to_tk
from mqt.bench import get_one_benchmark


def test_qiskit_native_gates():
    assert qiskit_plugin.get_ibm_native_gates() == ["rz", "sx", "x", "cx"]
    assert qiskit_plugin.get_rigetti_native_gates() == ["rx", "rz", "cz"]
    assert qiskit_plugin.get_ionq_native_gates() == ["rxx", "rz", "ry", "rx"]
    assert qiskit_plugin.get_oqc_native_gates() == ["rz", "sx", "x", "ecr"]


def test_qiskit_gate_counts():
    qc = get_one_benchmark("dj", 1, 3)
    get_ibm_montreal_qc = qiskit_plugin.get_ibm_montreal_qc(qc, 2)
    assert not get_ibm_montreal_qc is None
    ibm_washington_qc = qiskit_plugin.get_ibm_washington_qc(qc, 2)
    assert not ibm_washington_qc is None
    get_ionq_qc = qiskit_plugin.get_ionq_qc(qc, 2)
    assert not get_ionq_qc is None
    get_rigetti_qc = qiskit_plugin.get_rigetti_qc(qc, 2)
    assert not get_rigetti_qc is None
    get_oqc_qc = qiskit_plugin.get_oqc_qc(qc, 2)
    assert not get_oqc_qc is None

    qc = get_one_benchmark("dj", 1, 13)
    get_ibm_montreal_qc = qiskit_plugin.get_ibm_montreal_qc(qc, 2)
    assert not get_ibm_montreal_qc is None
    ibm_washington_qc = qiskit_plugin.get_ibm_washington_qc(qc, 2)
    assert not ibm_washington_qc is None
    get_ionq_qc = qiskit_plugin.get_ionq_qc(qc, 2)
    assert get_ionq_qc is None
    get_rigetti_qc = qiskit_plugin.get_rigetti_qc(qc, 2)
    assert not get_rigetti_qc is None
    get_oqc_qc = qiskit_plugin.get_oqc_qc(qc, 2)
    assert get_oqc_qc is None


def test_tket_gate_counts():
    qc = get_one_benchmark("dj", 1, 3)
    qc_tket = qiskit_to_tk(qc)
    get_ibm_montreal_qc = pytket_plugin.get_ibm_montreal_qc(qc_tket, lineplacement=True)
    assert not get_ibm_montreal_qc is None
    get_ibm_montreal_qc = pytket_plugin.get_ibm_montreal_qc(
        qc_tket, lineplacement=False
    )
    assert not get_ibm_montreal_qc is None
    ibm_washington_qc = pytket_plugin.get_ibm_washington_qc(qc_tket, lineplacement=True)
    assert not ibm_washington_qc is None
    ibm_washington_qc = pytket_plugin.get_ibm_washington_qc(
        qc_tket, lineplacement=False
    )
    assert not ibm_washington_qc is None
    get_ionq_qc = pytket_plugin.get_ionq_qc(qc_tket)
    assert not get_ionq_qc is None
    get_rigetti_qc = pytket_plugin.get_rigetti_qc(qc_tket, lineplacement=True)
    assert not get_rigetti_qc is None
    get_rigetti_qc = pytket_plugin.get_rigetti_qc(qc_tket, lineplacement=False)
    assert not get_rigetti_qc is None
    get_oqc_qc = pytket_plugin.get_oqc_qc(qc_tket, lineplacement=True)
    assert not get_oqc_qc is None
    get_oqc_qc = pytket_plugin.get_oqc_qc(qc_tket, lineplacement=False)
    assert not get_oqc_qc is None

    qc = get_one_benchmark("dj", 1, 13)
    qc_tket = qiskit_to_tk(qc)
    get_ibm_montreal_qc = pytket_plugin.get_ibm_montreal_qc(qc_tket, lineplacement=True)
    assert not get_ibm_montreal_qc is None
    get_ibm_montreal_qc = pytket_plugin.get_ibm_montreal_qc(
        qc_tket, lineplacement=False
    )
    assert not get_ibm_montreal_qc is None
    ibm_washington_qc = pytket_plugin.get_ibm_washington_qc(qc_tket, lineplacement=True)
    assert not ibm_washington_qc is None
    ibm_washington_qc = pytket_plugin.get_ibm_washington_qc(
        qc_tket, lineplacement=False
    )
    assert not ibm_washington_qc is None
    get_ionq_qc = pytket_plugin.get_ionq_qc(qc_tket)
    assert get_ionq_qc is None
    get_rigetti_qc = pytket_plugin.get_rigetti_qc(qc_tket, lineplacement=True)
    assert not get_rigetti_qc is None
    get_rigetti_qc = pytket_plugin.get_rigetti_qc(qc_tket, lineplacement=False)
    assert not get_rigetti_qc is None
    get_oqc_qc = pytket_plugin.get_oqc_qc(qc_tket, lineplacement=True)
    assert get_oqc_qc is None
    get_oqc_qc = pytket_plugin.get_oqc_qc(qc_tket, lineplacement=False)
    assert get_oqc_qc is None


def test_get_qiskit_qc():
    qc = get_one_benchmark("ghz", 1, 5)
    res = qiskit_plugin.save_qiskit_compiled_circuits(qc, 2, 10, "ghz_indep_5.qasm")
    assert res


def test_get_tket_qc():
    qc = get_one_benchmark("ghz", 1, 5)
    qc_tket = qiskit_to_tk(qc)
    res = pytket_plugin.save_tket_compiled_circuits(
        qc_tket, True, 10, "ghz_indep_5.qasm"
    )
    assert res
