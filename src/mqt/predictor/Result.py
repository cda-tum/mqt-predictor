from __future__ import annotations

from typing import TYPE_CHECKING

from mqt.predictor import reward

if TYPE_CHECKING:
    from qiskit import QuantumCircuit


class Result:
    """
    The Result class is used to store the results of a compilation.

    Attributes:
        benchmark (str): The path to the benchmark to be compiled.
        compiler (str): The setup used for compilation. Either 'mqt-predictor', 'qiskit_o3' or 'tket'.
        compilation_time (float): The time it took to compile the benchmark.
        compiled_qc (QuantumCircuit | None): The compiled quantum circuit. If compilation failed, None is returned.
        device (str): The device used for compilation.

    """

    def __init__(
        self,
        benchmark: str,
        compiler: str,
        compilation_time: float,
        compiled_qc: QuantumCircuit | None,
        device: str,
    ) -> None:
        if compiled_qc is not None:
            rew_fid = reward.expected_fidelity(compiled_qc, device)
            rew_crit_depth = reward.crit_depth(compiled_qc)
        else:
            rew_fid = -1.0
            rew_crit_depth = -1.0

        self.benchmark = benchmark
        self.compiler = compiler
        self.compilation_time = compilation_time
        self.fidelity = rew_fid
        self.critical_depth = rew_crit_depth

    def get_dict(self) -> dict[str, float]:
        """Returns the results as a dictionary."""

        return {
            self.compiler + "_" + "time": self.compilation_time,
            self.compiler + "_" + "expected_fidelity": self.fidelity,
            self.compiler + "_" + "critical_depth": self.critical_depth,
        }
