from __future__ import annotations

import pytest
from qiskit import QuantumCircuit

from mqt.predictor import reward


def test__false_inputs() -> None:
    with pytest.raises(ValueError):
        reward.expected_fidelity(QuantumCircuit(), "test_device")

    with pytest.raises(ValueError):
        reward.calc_expected_fidelity_ibm(QuantumCircuit(), "test_device")

    with pytest.raises(ValueError):
        reward.calc_expected_fidelity_ionq(QuantumCircuit(), "test_device")
