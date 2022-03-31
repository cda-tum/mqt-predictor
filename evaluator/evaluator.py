from qiskit_plugin import *
from pytket_plugin import *

from utils import get_openqasm_gates
from mqt.bench import benchmark_generator

from pytket.extensions.qiskit import qiskit_to_tk

import numpy as np
import argparse
from datetime import datetime


def dict_to_featurevector(gate_dict):
    openqasm_gates_list = get_openqasm_gates()
    res_dct = {openqasm_gates_list[i] for i in range(0, len(openqasm_gates_list))}
    res_dct = dict.fromkeys(res_dct, 0)
    for key, val in dict(gate_dict).items():
        if not key in res_dct:
            print(key, "gate not found in openQASM 2.0 gateset")
        else:
            res_dct[key] = val
    return res_dct


def create_training_data(min_qubit: int, max_qubit: int, stepsize: int = 1):
    benchmarks = [
        "dj",
        "grover-noancilla",
        "grover-v-chain",
        "ghz",
        "graphstate",
        "qft",
        "qftentangled",
        "qpeexact",
        "qpeinexact",
        "qwalk-noancilla",
        "qwalk-v-chain",
        "realamprandom",
        "su2random",
        "twolocalrandom",
        "vqe",
        "wstate",
    ]
    res = []
    for benchmark in benchmarks:
        for num_qubits in range(min_qubit, max_qubit, stepsize):
            if (
                "noancilla" in benchmark
                and num_qubits > 12
                or "v-chain" in benchmark
                and num_qubits > 12
            ):
                break
            print(benchmark, num_qubits)
            qc = benchmark_generator.get_one_benchmark(benchmark, 1, num_qubits)

            qasm_qc = qc.qasm()
            qc = QuantumCircuit.from_qasm_str(qasm_qc)
            qiskit_score = get_qiskit_scores(qc)
            try:
                qc_tket = qiskit_to_tk(qc)
                ops_list = qc.count_ops()
                tket_scores = get_tket_scores(qc_tket)
                best_arch = np.argmin(tket_scores + qiskit_score)
                res.append((ops_list, best_arch, num_qubits))
            except Exception as e:
                print("fail: ", e)
                # qc_tket = qasm.circuit_from_qasm_str(qc.qasm())
                # tket_scores = get_tket_scores(qc_tket)
        # else:

    training_data = []
    for elem in res:
        tmp = dict_to_featurevector(elem[0])
        tmp["num_qubits"] = elem[2]
        training_data.append((list(tmp.values()), elem[1]))

    x_train, y_train = zip(*training_data)
    current_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    with open(current_date + "_X_train.csv", "w") as FOUT:
        np.savetxt(FOUT, x_train)
    with open(current_date + "_y_train.csv", "w") as FOUT:
        np.savetxt(FOUT, y_train)

    return res


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create Training Data")
    parser.add_argument(
        "--min", type=int, default=3,
    )
    parser.add_argument(
        "--max", type=int, default=20,
    )
    parser.add_argument("--step", type=int, default=3)

    args = parser.parse_args()
    characteristics = create_training_data(args.min, args.max, args.step)

    print("Done")
