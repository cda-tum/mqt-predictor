from evaluator.src.qiskit_plugin import *
from evaluator.src.pytket_plugin import *

from evaluator.src.utils import get_openqasm_gates
from mqt.bench import benchmark_generator

from pytket.extensions.qiskit import qiskit_to_tk

import numpy as np
import signal

import json

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


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


def dict_to_featurevector(gate_dict, num_qubits):
    openqasm_gates_list = get_openqasm_gates()
    res_dct = {openqasm_gates_list[i] for i in range(0, len(openqasm_gates_list))}
    res_dct = dict.fromkeys(res_dct, 0)
    for key, val in dict(gate_dict).items():
        if not key in res_dct:
            print(key, "gate not found in openQASM 2.0 gateset")
        else:
            res_dct[key] = val

    res_dct["num_qubits"] = num_qubits
    return res_dct


def create_gate_lists(
    min_qubit: int, max_qubit: int, stepsize: int = 1, timeout: int = 10
):
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
        "qaoa",
        "portfoliovqe",
        "portfolioqaoa",
        "qgan",
    ]
    res = []
    for benchmark in benchmarks:
        for num_qubits in range(min_qubit, max_qubit, stepsize):
            print(benchmark, num_qubits)
            qc = timeout_watcher(
                benchmark_generator.get_one_benchmark,
                [benchmark, 1, num_qubits],
                timeout,
            )
            if not qc:
                break
            actual_num_qubits = qc.num_qubits
            qasm_qc = qc.qasm()
            qc = QuantumCircuit.from_qasm_str(qasm_qc)
            qiskit_gates = timeout_watcher(get_qiskit_gates, [qc], timeout)
            if not qiskit_gates:
                break
            try:
                qc_tket = qiskit_to_tk(qc)
                ops_list = qc.count_ops()
                feature_vector = dict_to_featurevector(ops_list, actual_num_qubits)
                tket_gates = timeout_watcher(get_tket_gates, [qc_tket], timeout)
                if not tket_gates:
                    break
                benchmark_name = benchmark + "_" + str(num_qubits)
                res.append(
                    (
                        benchmark,
                        feature_vector,
                        qiskit_gates + tket_gates,
                        qasm_qc,
                        benchmark_name,
                    )
                )
            except Exception as e:
                print("fail: ", e)

    jsonString = json.dumps(res, indent=4, sort_keys=True)
    with open("json_data.json", "w") as outfile:
        outfile.write(jsonString)
    return


def extract_training_data_from_json(json_path: str = "json_data.json"):
    with open(json_path, "r") as f:
        data = json.load(f)
    training_data = []
    qasm_list = []
    name_list = []

    for benchmark in data:
        scores = []
        num_qubits = benchmark[1]["num_qubits"]
        # Qiskit Scores
        for elem in benchmark[2][1]:
            if elem[0] is None:
                score = get_width_penalty()
            else:
                score = calc_score_from_gates_list(
                    elem[0], get_backend_information(elem[1])
                )
            scores.append(score)
        # Tket Scores
        for elem in benchmark[2][3]:
            if elem[0] is None:
                score = get_width_penalty()
            else:
                score = calc_score_from_gates_list(
                    elem[0], get_backend_information(elem[1])
                )
            scores.append(score)

        training_data.append((list(benchmark[1].values()), np.argmin(scores)))
        qasm_list.append(benchmark[3])
        name_list.append(benchmark[4])
        machines = get_machines()
        # print(num_qubits, machines[np.argmin(scores)])
    return (training_data, qasm_list, name_list)


def train_simple_ml_model(
    X, y, show_test_pred=False, eval_pred=True, qasm_list=None, name_list=None
):
    # X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.30, random_state=40)

    X, y, indices = np.array(X), np.array(y), np.array(range(len(y)))
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        X, y, indices, test_size=0.3, random_state=42
    )
    indices_train, indices_test

    model = Sequential()
    model.add(Dense(500, activation="relu", input_dim=43))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(10, activation="softmax"))

    # Compile the model
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(X_train, y_train, epochs=100)

    pred_train = model.predict(X_train)
    scores = model.evaluate(X_train, y_train, verbose=0)
    print(
        "Accuracy on training data: {}% \n Error on training data: {}".format(
            scores[1], 1 - scores[1]
        )
    )

    pred_test = model.predict(X_test)
    scores2 = model.evaluate(X_test, y_test, verbose=True)
    print(
        "Accuracy on test data: {}% \n Error on test data: {}".format(
            scores2[1], 1 - scores2[1]
        )
    )

    if show_test_pred:
        names_filtered = [name_list[i] for i in indices_test]
        check_test_predictions(pred_test, y_test, names_filtered)

    if eval_pred:
        # print("Indices: ", indices_test, [name_list[i] for i in indices_test], [qasm_list[i] for i in indices_test])
        eval_y_pred(
            qasm_qc_list=[qasm_list[i] for i in indices_test],
            y_pred=y_test,
            name_list=[name_list[i] for i in indices_test],
        )

    return model


def check_test_predictions(pred_test, y_test, benchmark_names):
    machines = get_machines()
    for i, pred in enumerate(pred_test):
        if np.argmax(pred) != y_test[i]:
            print(machines[np.argmax(pred)], machines[y_test[i]], benchmark_names[i])

    return


def eval_y_pred(qasm_qc_list, y_pred, name_list):
    res = []
    circuit_names = []

    plt.figure(figsize=(17, 6))

    for i, qasm_qc in enumerate(qasm_qc_list):

        qc_qiskit = QuantumCircuit.from_qasm_str(qasm_qc)
        qiskit_gates = get_qiskit_gates(qc_qiskit)[1]
        tmp_res = []
        for gate_list, machine in qiskit_gates:
            if gate_list is None:
                tmp_res.append(get_width_penalty())
            else:
                tmp_res.append(
                    calc_score_from_gates_list(
                        gate_list, get_backend_information(machine)
                    )
                )

        qc_tket = qiskit_to_tk(qc_qiskit)
        tket_gates = get_tket_gates(qc_tket)[1]
        for gate_list, machine in tket_gates:
            if gate_list is None:
                tmp_res.append(get_width_penalty())
            else:
                tmp_res.append(
                    calc_score_from_gates_list(
                        gate_list, get_backend_information(machine)
                    )
                )

        res.append(tmp_res)
        circuit_names.append(name_list[i])
        machines = get_machines()

        for j in range(10):
            plt.plot(i, tmp_res[j], ".", alpha=0.5, label=machines[j])
        plt.plot(i, tmp_res[y_pred[i]], "ko")
        plt.xlabel(get_machines())

        if machines[np.argmin(tmp_res)] != machines[y_pred[i]]:
            diff = tmp_res[y_pred[i]] - tmp_res[np.argmin(tmp_res)]
            print(
                machines[np.argmin(tmp_res)],
                machines[y_pred[i]],
                name_list[i],
                diff,
                tmp_res,
            )

    plt.title("Evaluation: Compilation Flow Prediction")
    plt.xticks(range(len(y_pred)), circuit_names, rotation=90)
    plt.xlabel("Unseen Benchmarks")
    plt.ylabel("Actual Score")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="upper right")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("y_pred_eval")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create Training Data")
    parser.add_argument(
        "--min",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--max",
        type=int,
        default=20,
    )
    parser.add_argument("--step", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=10)
    parser.parse_args()

    args = parser.parse_args()
    create_gate_lists(args.min, args.max, args.step, args.timeout)

    training_data, qasm_list, name_list = extract_training_data_from_json(
        "json_data.json"
    )
    X, y = zip(*training_data)
    train_simple_ml_model(X, y, True, True, qasm_list, name_list)

    # training_data, qasm_list, name_list = extract_training_data_from_json(
    #     "json_data_big.json"
    # )
    # print(qasm_list, name_list)
    # X, y = zip(*training_data)
    # train_simple_ml_model(X, y, True)
    # print("Done")
