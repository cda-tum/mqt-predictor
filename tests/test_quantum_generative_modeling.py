from __future__ import annotations

import quantum_generative_modeling as qgm
from qiskit.compiler import transpile
from qiskit.providers.aer import AerSimulator
from qiskit.providers.fake_provider import FakeNairobiV2, FakeQuitoV2


def test_generate_circuit() -> None:
    qc = qgm.generate_circuit(4, 1, 2)
    assert qc.num_qubits == 4
    assert qc.depth() == 7
    assert qc.num_clbits == 4
    assert len(qc.data) == 22


def test_main() -> None:
    qc_4 = qgm.generate_circuit(4)
    fake_backend_4 = FakeQuitoV2()
    qc_6 = qgm.generate_circuit(6)
    fake_backend_6 = FakeNairobiV2()

    for qc, fake_backend in [(qc_4, fake_backend_4), (qc_6, fake_backend_6)]:
        qcbm = qgm.QCBM(n_qubits=qc.num_qubits)
        backend = AerSimulator.from_backend(fake_backend)
        compiled_circuit = transpile(qc, backend=fake_backend, optimization_level=3)
        best_kl, _ = qcbm.train(circuit=compiled_circuit.copy(), backend=backend)
        assert 0 < best_kl < 1
