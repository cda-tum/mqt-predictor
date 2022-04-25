from evaluator.src.qiskit_plugin import *
from evaluator.src.pytket_plugin import *

from evaluator.src.utils import get_openqasm_gates
from mqt.bench import benchmark_generator

from pytket.extensions.qiskit import qiskit_to_tk

import numpy as np
import signal

import json
import os
import argparse
import csv

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

from natsort import natsorted
import glob

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
        if key in res_dct:
            res_dct[key] = val

    res_dct["num_qubits"] = num_qubits
    return res_dct


def create_gate_lists_from_folder(folder_path: str = "./qasm_files", timeout: int = 10):

    res = []

    filelist = glob.glob(folder_path)

    dictionary = {}
    for subdir, dirs, files in os.walk(folder_path):
        for file in natsorted(files):
            if "qasm" in file:
                key = file.split("_")[
                    0
                ]  # The key is the first 16 characters of the file name
                group = dictionary.get(key, [])
                group.append(file)
                dictionary[key] = group

    print(dictionary.keys())
    for alg_class in dictionary:
        for benchmark in dictionary[alg_class]:
            filename = os.path.join(folder_path, benchmark)
            qc = QuantumCircuit.from_qasm_file(filename)

            num_qubits = qc.num_qubits
            print(benchmark)
            if not qc:
                continue
            actual_num_qubits = qc.num_qubits

            try:
                qiskit_gates = timeout_watcher(get_qiskit_gates, [qc], timeout)
                if not qiskit_gates:
                    break

                qc_tket = qiskit_to_tk(qc)
                tket_gates = timeout_watcher(get_tket_gates, [qc_tket], timeout)
                if not tket_gates:
                    break

                ops_list = qc.count_ops()
                feature_vector = dict_to_featurevector(ops_list, actual_num_qubits)
                benchmark_name = benchmark + "_" + str(num_qubits)
                res.append(
                    (
                        benchmark,
                        feature_vector,
                        qiskit_gates + tket_gates,
                        benchmark_name,
                    )
                )
            except Exception as e:
                print("fail: ", e)

    jsonString = json.dumps(res, indent=4, sort_keys=True)
    with open("json_data.json", "w") as outfile:
        outfile.write(jsonString)
    return


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
            try:

                qiskit_gates = timeout_watcher(get_qiskit_gates, [qc], timeout)
                if not qiskit_gates:
                    break

                qc_tket = qiskit_to_tk(qc)
                tket_gates = timeout_watcher(get_tket_gates, [qc_tket], timeout)
                if not tket_gates:
                    break

                ops_list = qc.count_ops()
                feature_vector = dict_to_featurevector(ops_list, actual_num_qubits)
                benchmark_name = benchmark + "_" + str(num_qubits)
                res.append(
                    (
                        benchmark,
                        feature_vector,
                        qiskit_gates + tket_gates,
                        benchmark_name,
                    )
                )
            except Exception as e:
                print("fail: ", e)

    jsonString = json.dumps(res, indent=4, sort_keys=True)
    with open("json_data.json", "w") as outfile:
        outfile.write(jsonString)
    return


def extract_training_data_from_json(
    json_path: str = "json_data.json", eval_fid: bool = True
):
    with open(json_path, "r") as f:
        data = json.load(f)
    training_data = []
    name_list = []
    scores_list = []

    for benchmark in data:
        scores = []
        num_qubits = benchmark[1]["num_qubits"]
        # Qiskit Scores
        for elem in benchmark[2][1]:
            if elem[0] is None:
                score = get_width_penalty()
            else:

                score = calc_score_from_gates_list(
                    elem[0], get_backend_information(elem[1]), num_qubits
                )

            scores.append(score)
        # Tket Scores
        for elem in benchmark[2][3]:
            if elem[0] is None:
                score = get_width_penalty()
            else:
                score = calc_score_from_gates_list(
                    elem[0], get_backend_information(elem[1]), num_qubits
                )

            scores.append(score)

        training_data.append((list(benchmark[1].values()), np.argmin(scores)))
        name_list.append(benchmark[3])
        scores_list.append(scores)

    return (training_data, name_list, scores_list)


def train_simple_ml_model(
    X, y, eval_pred=True, name_list=None, actual_scores_list=None
):

    X, y, indices = np.array(X), np.array(y), np.array(range(len(y)))
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        X, y, indices, test_size=0.3, random_state=42
    )
    indices_train, indices_test

    model = Sequential()
    model.add(Dense(500, activation="relu", input_dim=len(X[0])))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(10, activation="softmax"))

    # Compile the model
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(X_train, y_train, epochs=100)

    scores = model.evaluate(X_train, y_train, verbose=0)
    print(
        "Accuracy on training data: {}% \n Error on training data: {}".format(
            scores[1], 1 - scores[1]
        )
    )

    scores2 = model.evaluate(X_test, y_test, verbose=1)
    print(
        "Accuracy on test data: {}% \n Error on test data: {}".format(
            scores2[1], 1 - scores2[1]
        )
    )

    pred_test = model.predict(X_test)
    if eval_pred:
        eval_y_pred(
            y_predicted=pred_test,
            y_actual=y_test,
            names_list=[name_list[i] for i in indices_test],
            scores_filtered=[actual_scores_list[i] for i in indices_test],
        )

    return model


def check_test_predictions(pred_test, y_test, benchmark_names):
    machines = get_machines()
    for i, pred in enumerate(pred_test):
        if np.argmax(pred) != y_test[i]:
            print(machines[np.argmax(pred)], machines[y_test[i]], benchmark_names[i])

    return


def eval_y_pred(y_predicted, y_actual, names_list, scores_filtered):
    circuit_names = []

    all_rows = []
    all_rows.append(
        [
            "Benchmark",
            "Best Score",
            "MQT Predictor",
            "Best Machine",
            "MQT Predictor",
            "Overhead",
        ]
    )

    plt.figure(figsize=(17, 6))
    print("len(y_predicted)", len(y_predicted))
    for i in range(len(y_predicted)):
        y_predicted_instance = np.argmax(y_predicted[i])
        if y_predicted_instance != y_actual[i]:
            row = []
            tmp_res = scores_filtered[i]
            circuit_names.append(names_list[i])
            machines = get_machines()

            comp_val = tmp_res[y_predicted_instance] / tmp_res[y_actual[i]]
            row.append(names_list[i])
            row.append(np.round(np.min(tmp_res), 2))
            row.append(np.round(tmp_res[y_predicted_instance], 2))
            row.append(y_actual[i])
            row.append(y_predicted_instance)
            row.append(np.round(comp_val - 1.00, 2))
            all_rows.append(row)

            for j in range(10):
                plt.plot(len(circuit_names), tmp_res[j], ".", alpha=0.5, label=machines[j])
            plt.plot(len(circuit_names), tmp_res[y_predicted_instance], "ko", label="MQTPredictor")
            plt.xlabel(get_machines())

            if machines[np.argmin(tmp_res)] != machines[y_predicted_instance]:
                assert np.argmin(tmp_res) == y_actual[i]
                diff = tmp_res[y_predicted_instance] - tmp_res[np.argmin(tmp_res)]
                print(
                    names_list[i],
                    " predicted: ",
                    y_predicted_instance,
                    " should be: ",
                    y_actual[i],
                    " diff: ",
                    diff,
                )

    plt.title("Evaluation: Compilation Flow Prediction")
    plt.xticks(range(len(circuit_names)), circuit_names, rotation=90)
    plt.xlabel("Unseen Benchmarks")
    plt.ylabel("Actual Score")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="upper right")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("y_pred_eval")


    with open("results.csv", "w", encoding="UTF8", newline="") as f:
        writer = csv.writer(f)

        for row in all_rows:
            writer.writerow(row)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create Training Data")
    # parser.add_argument(
    #     "--min",
    #     type=int,
    #     default=3,
    # )
    # parser.add_argument(
    #     "--max",
    #     type=int,
    #     default=20,
    # )
    # parser.add_argument("--step", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=10)
    # parser.parse_args()
    #
    args = parser.parse_args()
    # create_gate_lists(args.min, args.max, args.step, args.timeout)

    create_gate_lists_from_folder(timeout=args.timeout)

    # training_data, name_list, scores = extract_training_data_from_json("json_data.json")
    # X, y = zip(*training_data)
    # train_simple_ml_model(X, y, True, name_list, scores)

    # create_gate_lists_from_folder(timeout=30)
    # training_data, qasm_list, name_list = extract_training_data_from_json(
    #     "json_data_big.json"
    # )
    # print(qasm_list, name_list)
    # X, y = zip(*training_data)
    # train_simple_ml_model(X, y, True)
    # print("Done")
