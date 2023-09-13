from __future__ import annotations

from typing import TYPE_CHECKING

from mqt.predictor import reward, rl

if TYPE_CHECKING:
    from qiskit import QuantumCircuit


class Result:
    """
    The Result class is used to store the results of a compilation.

    Attributes:
        benchmark (str): The path to the benchmark to be compiled.
        used_setup (str): The setup used for compilation. Either 'MQTPredictor', 'qiskit_o3' or 'tket'.
        duration (float): The time it took to compile the benchmark.
        qc (QuantumCircuit | None): The compiled quantum circuit. If compilation failed, None is returned.
        device (str): The device used for compilation.

    """

    def __init__(
        self,
        benchmark: str,
        used_setup: str,
        duration: float,
        qc: QuantumCircuit | None,
        device: str,
    ) -> None:
        if qc is not None:
            rew_fid = reward.expected_fidelity(qc, device)
            rew_crit_depth = reward.crit_depth(qc)
        else:
            rew_fid = -1
            rew_crit_depth = -1

        self.benchmark: str = benchmark
        self.used_setup: str = used_setup
        self.time: float = duration
        self.fidelity: float = rew_fid
        self.critical_depth: float = rew_crit_depth

    def get_dict(self) -> dict[str, float]:
        """Returns the results as a dictionary."""

        return {
            self.used_setup + "_" + "time": self.time,
            self.used_setup + "_" + "fidelity": self.fidelity,
            self.used_setup + "_" + "critical_depth": self.critical_depth,
        }


class MQTDurationResult:
    """
    The MQTDurationResult class is used to store the results of a compilation.
    """

    def __init__(
        self,
        benchmark: str,
        figure_of_merit: str,
        res_mqt: list[float],
    ) -> None:
        self.benchmark: str = benchmark
        self.res_mqt = res_mqt
        self.figure_of_merit = figure_of_merit

    def get_dict(self) -> dict[str, float | str]:
        """Returns the results as a dictionary."""
        overall: dict[str, float | str] = {"benchmark": self.benchmark}
        for i, dev in enumerate(rl.helper.get_devices()):
            overall.update(
                {
                    "mqt_duration_" + self.figure_of_merit + "_" + dev["name"]: self.res_mqt[i],
                }
            )

        return overall
