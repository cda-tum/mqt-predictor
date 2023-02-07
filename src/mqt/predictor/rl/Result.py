from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypedDict

from mqt.predictor import reward

if TYPE_CHECKING:
    from pathlib import Path

    from mqt.predictor.devices import Device
    from mqt.predictor.rl import RewardFunction
    from qiskit import QuantumCircuit


Setup = Literal["RL", "qiskit", "tket"]


class ResultDict(TypedDict):
    benchmark: str
    num_qubits: int
    runtime: float
    fidelity: float
    critical_depth: float
    gate_ratio: float
    mix: float
    setup: Setup
    reward_function: RewardFunction | None


class Result:
    """
    Class to calculate and store the result of a compiler for a given benchmark.
    """

    def __init__(
        self,
        benchmark: Path,
        runtime: float,
        qc: QuantumCircuit,
        device: Device,
        setup: Setup,
        reward_function: RewardFunction | None = None,
    ) -> None:
        self.benchmark: str = str(benchmark.stem).replace("_", " ").split(" ")[0]
        self.num_qubits: int = int(str(benchmark.stem).replace("_", " ").split(" ")[-1])
        self.runtime: float = runtime

        self.fidelity: float = reward.expected_fidelity(qc, device)
        self.critical_depth: float = reward.crit_depth(qc)
        self.gate_ratio: float = reward.gate_ratio(qc)
        self.mix: float = reward.mix(qc, device)

        self.setup: Setup = setup
        self.reward_function: RewardFunction | None = reward_function

    def get_dict(self) -> ResultDict:
        return {
            "benchmark": self.benchmark,
            "num_qubits": self.num_qubits,
            "runtime": self.runtime,
            "fidelity": self.fidelity,
            "critical_depth": self.critical_depth,
            "gate_ratio": self.gate_ratio,
            "mix": self.mix,
            "setup": self.setup,
            "reward_function": self.reward_function,
        }
