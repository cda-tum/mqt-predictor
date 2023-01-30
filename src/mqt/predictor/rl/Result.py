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

    def __init__(
        self,
        benchmark,
        used_setup,
        time,
        fidelity,
        depth,
        gate_ratio,
        mix,
    ):
        self.benchmark = benchmark
        self.used_setup = used_setup
        self.time = time
        self.fidelity = fidelity
        self.depth = depth
        self.gate_ratio = gate_ratio
        self.mix = mix

    def __str__(self):
        return (
            f"Result(benchmark={self.benchmark}, used_setup={self.used_setup}, time={self.time}, "
            f"fidelity={self.fidelity}, depth={self.depth}, gate_ratio={self.gate_ratio}, "
            f"mix={self.mix})"
        )

    def __repr__(self):
        return (
            f"Result(benchmark={self.benchmark}, used_setup={self.used_setup}, time={self.time}, "
            f"fidelity={self.fidelity}, depth={self.depth}, gate_ratio={self.gate_ratio}, "
            f"mix={self.mix})"
        )

    def get_result(self):
        return {
            self.used_setup + "_" + "time": self.time,
            self.used_setup + "_" + "fidelity": self.fidelity,
            self.used_setup + "_" + "depth": self.depth,
            self.used_setup + "_" + "gate_ratio": self.gate_ratio,
            self.used_setup + "_" + "mix": self.mix,
        }

    def __eq__(self, other):
        return (
            self.benchmark == other.benchmark
            and self.reward == other.reward
            and self.time == other.time
            and self.fidelity == other.fidelity
            and self.depth == other.depth
            and self.gate_ratio == other.gate_ratio
            and self.mix == other.mix
        )
