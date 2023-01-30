from dataclasses import dataclass


@dataclass
class Result:
    """
    Class to store the result of a compiler for a given benchmark.

    Attributes
    benchmark: str - name of the benchmark
    used_setup: str - name of the used setup
    time: float - time needed to compile the circuit
    fidelity: float - fidelity reward of the compiled circuit
    depth: float - depth reward of the compiled circuit
    gate_ratio: float - gate ratio reward of the compiled circuit
    mix: float - mix reward of the compiled circuit

    """

    benchmark: str
    used_setup: str
    time: float
    fidelity: float
    depth: int
    gate_ratio: float
    mix: float

    def get_dict(self):
        return {
            self.used_setup + "_" + "time": self.time,
            self.used_setup + "_" + "fidelity": self.fidelity,
            self.used_setup + "_" + "critical_depth": self.depth,
            self.used_setup + "_" + "gate_ratio": self.gate_ratio,
            self.used_setup + "_" + "mix": self.mix,
        }
