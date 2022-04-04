from qiskit_plugin import *
from pytket_plugin import *

from utils import get_openqasm_gates
from mqt.bench import benchmark_generator

from pytket.extensions.qiskit import qiskit_to_tk

import numpy as np
import argparse
import signal

import json

from datetime import datetime

def timeout_watcher(func, args, timeout):
    class TimeoutException(Exception):  # Custom exception class
        pass

    def timeout_handler(signum, frame):  # Custom signal handler
        raise TimeoutException

    # Change the behavior of SIGALRM
    signal.signal(signal.SIGALRM, timeout_handler)

    signal.alarm(timeout)
    try:
        res = func(*args)
    except TimeoutException:
        print("Calculation/Generation exceeded timeout limit for ", func, args[1:])
        return False
    except Exception as e:
        print("Something else went wrong: ", e)
        return False
    else:
        # Reset the alarm
        signal.alarm(0)

    return res

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


def create_training_data(min_qubit: int, max_qubit: int, stepsize: int = 1, timeout: int = 10):
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
        "vqe",
        "qaoa",
        "portfoliovqe",
        "portfolioqaoa",
        "qgan"
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
            qc = timeout_watcher(benchmark_generator.get_one_benchmark, [benchmark, 1, num_qubits], timeout)

            if not qc:
                break
            qasm_qc = qc.qasm()
            qc = QuantumCircuit.from_qasm_str(qasm_qc)
            qiskit_score = timeout_watcher(get_qiskit_scores, [qc], timeout)
            if not qiskit_score:
                break
            try:
                qc_tket = qiskit_to_tk(qc)
                ops_list = qc.count_ops()
                tket_scores = timeout_watcher(get_tket_scores, [qc_tket], timeout)
                if not tket_scores:
                    break
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

def create_gate_lists(min_qubit: int, max_qubit: int, stepsize: int = 1, timeout: int = 10):
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
        "vqe",
        "qaoa",
        "portfoliovqe",
        "portfolioqaoa",
        "qgan"
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
            qc = timeout_watcher(benchmark_generator.get_one_benchmark, [benchmark, 1, num_qubits], timeout)

            if not qc:
                break
            qasm_qc = qc.qasm()
            qc = QuantumCircuit.from_qasm_str(qasm_qc)
            qiskit_gates = timeout_watcher(get_qiskit_gates, [qc], timeout)
            if not qiskit_gates:
                break
            try:
                qc_tket = qiskit_to_tk(qc)
                ops_list = qc.count_ops()
                tket_gates = timeout_watcher(get_tket_gates, [qc_tket], timeout)
                if not tket_gates:
                    break
                res.append((benchmark, num_qubits, ops_list, qiskit_gates+tket_gates))
            except Exception as e:
                print("fail: ", e)


    jsonString = json.dumps(res, indent=4, sort_keys=True)
    with open('json_data.json', 'w') as outfile:
        outfile.write(jsonString)
    return




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create Training Data")
    parser.add_argument(
        "--min", type=int, default=3,
    )
    parser.add_argument(
        "--max", type=int, default=20,
    )
    parser.add_argument("--step", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=10)

    args = parser.parse_args()
    #characteristics = create_training_data(args.min, args.max, args.step, args.timeout)
    data = create_gate_lists(args.min, args.max, args.step, args.timeout)

    print("Done")
