from mqt.predictor import reward
from qiskit import QuantumCircuit
import pytest

def test__false_inputs() -> None:
    with pytest.raises(ValueError):
        reward.expected_fidelity(QuantumCircuit(), "test_device")

    with pytest.raises(ValueError):
        reward.calc_expected_fidelity_ibm(QuantumCircuit(), "test_device")

    with pytest.raises(ValueError):
        reward.calc_expected_fidelity_ionq(QuantumCircuit(), "test_device")



