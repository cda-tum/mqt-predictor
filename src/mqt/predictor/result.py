"""The Result class is used to store the results of a compilation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mqt.predictor import reward
from mqt.predictor.hellinger import hellinger_model_available

if TYPE_CHECKING:
    from qiskit import QuantumCircuit

    from mqt.bench.devices import Device


class Result:
    """The Result class is used to store the results of a compilation.

    Attributes:
        compiler: The setup used for compilation. Either 'mqt-predictor_<figure_of_merit>', 'qiskit' or 'tket'. For the two latter, also the device name is appended.
        compilation_time: The time it took to compile the benchmark.
        expected_fidelity: The expected fidelity of the compiled quantum circuit.
        critical_depth: The critical depth of the compiled quantum circuit.
        estimated_success_probability: The estimated success probability (ESP) of the compiled quantum circuit.
        estimated_hellinger_distance: The estimated Hellinger distance of the compiled quantum circuit.
    """

    """Constructor for the Result class.

    Arguments:
        compilation_setup: The setup used for compilation. Either 'mqt-predictor_<figure_of_merit>', 'qiskit' or 'tket'. For the two latter, also the device name is appended.
        compilation_time: The time it took to compile the benchmark.
        compiled_qc: The compiled quantum circuit.
        device: The device used for compilation.
    """

    def __init__(
        self,
        compilation_setup: str,
        compilation_time: float,
        compiled_qc: QuantumCircuit | None,
        device: Device,
    ) -> None:
        """Initializes the Result object."""
        if compiled_qc is not None:
            rew_fid = reward.expected_fidelity(compiled_qc, device)
            rew_crit_depth = reward.crit_depth(compiled_qc)
            rew_esp = (
                reward.estimated_success_probability(compiled_qc, device) if reward.esp_data_available(device) else -1.0
            )
            rew_hellinger = (
                reward.estimated_hellinger_distance(compiled_qc, device) if hellinger_model_available(device) else -1.0
            )
        else:
            rew_fid = -1.0
            rew_crit_depth = -1.0
            rew_esp = -1.0
            rew_hellinger = -1.0

        self.compiler = compilation_setup
        self.compilation_time = compilation_time
        self.expected_fidelity = rew_fid
        self.critical_depth = rew_crit_depth
        self.estimated_success_probability = rew_esp
        self.estimated_hellinger_distance = rew_hellinger

    def get_dict(self) -> dict[str, float]:
        """Returns the results as a dictionary."""
        return {
            self.compiler + "_" + "time": self.compilation_time,
            self.compiler + "_" + "expected_fidelity": self.expected_fidelity,
            self.compiler + "_" + "critical_depth": self.critical_depth,
            self.compiler + "_" + "estimated_success_probability": self.estimated_success_probability,
            self.compiler + "_" + "estimated_hellinger_distance": self.estimated_hellinger_distance,
        }
