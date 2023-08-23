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
        qc: QuantumCircuit,
        device: str,
    ) -> None:
        rew_fid = reward.expected_fidelity(qc, device)
        rew_crit_depth = reward.crit_depth(qc)

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


class LargeResult:
    def __init__(
        self,
        benchmark: str,
        res_qiskit: list[tuple[float, float, float]],
        res_tket: list[tuple[float, float, float]],
        res_MQT: list[tuple[float, float, float, float]],
    ) -> None:
        self.benchmark: str = benchmark
        self.res_qiskit = res_qiskit
        self.res_tket = res_tket
        self.res_MQT = res_MQT

    def get_dict(self) -> dict[str, float | str]:
        overall: dict[str, float | str] = {"benchmark": self.benchmark}
        for i, dev in enumerate(rl.helper.get_devices()):
            overall.update(
                {
                    "qiskit_fid_" + dev["name"]: self.res_qiskit[i][0],
                    "tket_fid_" + dev["name"]: self.res_tket[i][0],
                    "MQT_fid_" + dev["name"]: self.res_MQT[i][0],
                    "qiskit_dep_" + dev["name"]: self.res_qiskit[i][1],
                    "tket_dep_" + dev["name"]: self.res_tket[i][1],
                    "MQT_dep_" + dev["name"]: self.res_MQT[i][1],
                    "qiskit_duration_" + dev["name"]: self.res_qiskit[i][2],
                    "tket_duration_" + dev["name"]: self.res_tket[i][2],
                    "MQT_fid_duration_" + dev["name"]: self.res_MQT[i][2],
                    "MQT_dep_duration_" + dev["name"]: self.res_MQT[i][3],
                }
            )

        return overall
