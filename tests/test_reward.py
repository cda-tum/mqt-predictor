from __future__ import annotations

import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import FakeNairobiV2
from quantum_generative_modeling import generate_circuit

from mqt.predictor import reward


def test_KL() -> None:
    qc = generate_circuit(4)
    compiled_qc = transpile(qc, backend=FakeNairobiV2())
    res, _ = reward.KL(
        compiled_qc=compiled_qc,
        initial_gate_count=sum(compiled_qc.count_ops().values()),
        max_cx_count=100,
        num_initial_qubits=qc.num_qubits,
        device_name="ibm_nairobi",
    )
    assert 0 < res < 1


def test__false_inputs() -> None:
    with pytest.raises(ValueError, match="Device not supported"):
        reward.expected_fidelity(QuantumCircuit(), "test_device")

    with pytest.raises(ValueError, match="Device not supported"):
        reward.calc_expected_fidelity_ibm(QuantumCircuit(), "test_device")

    with pytest.raises(ValueError, match="Device not supported"):
        reward.calc_expected_fidelity_ionq(QuantumCircuit(), "test_device")
