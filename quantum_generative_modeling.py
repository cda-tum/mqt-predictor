from __future__ import annotations

import argparse
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from cma import CMAEvolutionStrategy
from joblib import Parallel, delayed
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.compiler import transpile
from qiskit.providers.aer import AerSimulator
from qiskit.providers.fake_provider import FakeMontrealV2, FakeNairobiV2, FakeQuitoV2
from scipy.special import binom


def generate_circuit(n_qubits: int, depth: int = 1, n_registers: int = 2) -> QuantumCircuit:
    circuit = QuantumCircuit(n_qubits, n_qubits)
    param_counter = 0

    n = n_qubits // n_registers

    for k in range(n):
        circuit.h(k)

    for j in range(n_registers - 1):
        for k in range(n):
            circuit.cx(k, k + n * (j + 1))

    shift = 0
    for _ in range(depth):
        for k in range(n):
            for j in range(n_registers):
                circuit.rz(Parameter(f"x_{param_counter:03d}"), j * n + k)
                param_counter += 1
                circuit.rx(Parameter(f"x_{param_counter:03d}"), j * n + k)
                param_counter += 1
                circuit.rz(Parameter(f"x_{param_counter:03d}"), j * n + k)
                param_counter += 1

        k = 3 * n + shift
        for i, j in combinations(range(n), 2):
            for m in range(n_registers):
                circuit.rxx(Parameter(f"x_{param_counter:03d}"), m * n + i, m * n + j)
                param_counter += 1

            k += 1
        shift += 3 * n + int(binom(n, 2))

    for k in range(n_qubits):
        circuit.measure(k, k)

    return circuit


class QCBM:
    """
    This module optimizes the parameters of quantum circuit using CMA-ES.
    This training method is referred to as quantum circuit born machine (QCBM).
    """

    def __init__(self, n_qubits: int, shots: int = 10000, population_size: int = 5, sigma: float = 0.5):
        if n_qubits <= 4:
            max_evaluations = 750
        elif n_qubits <= 6:
            max_evaluations = 1000
        else:
            max_evaluations = 2500
        self.target = self.get_target(n_qubits)
        self.n_shots = shots
        self.max_evaluations = max_evaluations
        self.population_size = population_size
        self.sigma = sigma
        self.n_qubits = n_qubits

    def train(self, circuit: QuantumCircuit, backend: fake_backend.FakeBackendV2) -> tuple[float, list[float]]:
        execute_circuit = self.get_execute_circuit(circuit, backend)
        evolution_data = []

        n_params = circuit.num_parameters
        x0 = (np.random.rand(n_params) - 0.5) * np.pi

        # Hyperparameters for the optimizer
        options = {
            "bounds": [n_params * [-np.pi], n_params * [np.pi]],
            "maxfevals": self.max_evaluations,
            "popsize": self.population_size,
            "verbose": -3,
            "tolfun": 0.0001,
        }

        # Instantiate the optimizer
        es = CMAEvolutionStrategy(x0, self.sigma, options)
        best_kl = float("inf")
        while not es.stop():
            # Sample five (corresponding to population size) possible solutions from covariance matrix
            solutions = es.ask()

            # Get five (corresponding to population size) pmfs (probability mass functions), that correspond to five different sets of parameter distributions
            pmfs_model = execute_circuit(solutions, self.n_shots)

            if pmfs_model is None:
                return 1.00
            # Calculate the loss function for each of the five pmfs
            loss_epoch = self.kl_divergence(pmfs_model.reshape([self.population_size, -1]))
            # Tell optimizer the loss function for each of the five pmfs, to determine the best performing sets of model parameters
            es.tell(solutions, loss_epoch)

            evolution_data.append(min(loss_epoch))
            if min(loss_epoch) < best_kl:
                best_kl = min(loss_epoch)

        # Return the lowest KL divergence (Figure of merit)
        return best_kl, evolution_data

    def kl_divergence(self, pmf_model: np.ndarray):
        # Loss function bwteen target and model distribution
        pmf_model[pmf_model == 0] = 1e-8
        return np.sum(self.target * np.log(self.target / pmf_model), axis=1)

    def get_execute_circuit(self, circuit_transpiled: QuantumCircuit, backend: fake_backend.FakeBackendV2):
        def execute_circuit(solutions, num_shots=None):
            # Execute the circuit and returns the probability mass function

            sample_dicts = Parallel(n_jobs=-1, verbose=0)(
                delayed(self.generate_result)(circuit_transpiled.assign_parameters(solution), num_shots, i, backend)
                for i, solution in enumerate(solutions)
            )
            if any(sample_dicts) is False:
                return None

            samples_dictionary = []
            for i in range(len(sample_dicts)):
                samples_dictionary.extend(sample_dicts[i].values())

            samples = []
            for result in samples_dictionary:
                target_iter = np.zeros(2**self.n_qubits)
                result_keys = list(result.keys())
                result_vals = list(result.values())
                target_iter[result_keys] = result_vals
                target_iter = np.asarray(target_iter)
                samples.append(target_iter)

            samples = np.asarray(samples)
            return samples / num_shots

        return execute_circuit

    @staticmethod
    def get_target(n_qubits):
        # Target distribution
        side_dimension = int(2 ** (n_qubits / 2))
        grid = np.zeros((side_dimension, side_dimension))
        for i in range(side_dimension):
            grid[i, i] = 1
            grid[i, side_dimension - 1 - i] = 1

        grid = 1 - grid
        grid += 1e-8
        grid /= grid.sum()

        return grid.flatten()

    def generate_result(self, qc: QuantumCircuit, num_shots: int, index: int, backend: fake_backend.FakeBackendV2):
        job = backend.run(qc, shots=num_shots)
        samples_dictionary = job.result().get_counts(qc).int_outcomes()
        return {index: samples_dictionary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QCBM")

    parser.add_argument("--num_qubits", type=int, default=4)
    parser.add_argument("--opt", type=int, default=3)
    parser.add_argument("--num_runs", type=int, default=10)
    args = parser.parse_args()
    n_qubits = args.num_qubits
    num_runs = args.num_runs
    opt_level = args.opt

    if n_qubits % 2 != 0:
        msg = "Number of qubits must be even"
        raise ValueError(msg)
    circuit = generate_circuit(n_qubits=n_qubits)
    qcbm = QCBM(n_qubits=n_qubits)

    if n_qubits <= 5:
        fake_backend = FakeQuitoV2()
    elif n_qubits <= 7:
        fake_backend = FakeNairobiV2()
    else:
        fake_backend = FakeMontrealV2()

    backend = AerSimulator.from_backend(fake_backend)

    print(
        f"Optimization Level {opt_level} with backend information for {n_qubits} qubits on {fake_backend.name} backend."
    )
    res = []
    all_eval_data = []
    for i in range(num_runs):
        print("Run", i)
        compiled_circuit = transpile(circuit, backend=fake_backend, optimization_level=opt_level)
        print(compiled_circuit.count_ops(), sum(compiled_circuit.count_ops().values()))
        best_KL, evolution_data = qcbm.train(circuit=compiled_circuit.copy(), backend=backend)
        all_eval_data.append(evolution_data)
        plt.show()
        res.append(best_KL)

    print(f"AVG KL={np.average(res)}, STD KL={np.std(res)}, BEST KL={np.min(res)}")
    np.savetxt(
        f"evaluations/results_application_aware_compilation/O{opt_level}/all_eval_data_{fake_backend.name}_{n_qubits}_qubits.txt",
        np.array(all_eval_data),
    )
