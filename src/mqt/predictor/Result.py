from __future__ import annotations

from typing import TYPE_CHECKING

from mqt.predictor import reward, rl

if TYPE_CHECKING:
    from qiskit import QuantumCircuit


class Result:
    """
    Class to calculate and store the result of a compiler for a given benchmark.

    Attributes
    benchmark: str - name of the benchmark
    used_setup: str - name of the used setup
    time: float - time needed to compile the circuit
    fidelity: float - fidelity reward of the compiled circuit
    depth: float - depth reward of the compiled circuit
    gate_ratio: float - gate ratio reward of the compiled circuit
    mix: float - mix reward of the compiled circuit

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
        return {
            self.used_setup + "_" + "time": self.time,
            self.used_setup + "_" + "fidelity": self.fidelity,
            self.used_setup + "_" + "critical_depth": self.critical_depth,
        }




class MQTDurationResult:
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
        overall: dict[str, float | str] = {"benchmark": self.benchmark}
        for i, dev in enumerate(rl.helper.get_devices()):
            overall.update(
                {
                    "mqt_duration_" + self.figure_of_merit + "_" + dev["name"]: self.res_mqt[i],
                }
            )

        return overall
