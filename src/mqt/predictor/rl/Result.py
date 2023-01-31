from qiskit import QuantumCircuit

from mqt.predictor import reward


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
    ):
        rew_fid = reward.expected_fidelity(qc, device)
        rew_crit_depth = reward.crit_depth(qc)
        rew_gate_ratio = reward.gate_ratio(qc)
        rew_mix = reward.mix(qc, device)

        self.benchmark = benchmark
        self.used_setup = used_setup
        self.time = duration
        self.fidelity = rew_fid
        self.critical_depth = rew_crit_depth
        self.gate_ratio = rew_gate_ratio
        self.mix = rew_mix

    def get_dict(self):
        return {
            self.used_setup + "_" + "time": self.time,
            self.used_setup + "_" + "fidelity": self.fidelity,
            self.used_setup + "_" + "critical_depth": self.critical_depth,
            self.used_setup + "_" + "gate_ratio": self.gate_ratio,
            self.used_setup + "_" + "mix": self.mix,
        }
