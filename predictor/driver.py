from predictor.src import qiskit_plugin, pytket_plugin, utils

from qiskit import QuantumCircuit
from pytket.extensions.qiskit import qiskit_to_tk

import numpy as np
import json
import os
import argparse
import csv

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

from natsort import natsorted
import glob
import matplotlib.pyplot as plt
from dtreeviz.trees import dtreeviz


class Predictor:
    _clf = None

    def create_gate_lists_from_folder(
        folder_path: str = "./qasm_files", timeout: int = 10
    ):
        """Method to create pre-process data to accelerate the training data generation afterwards. All .qasm files from
        the folder path are considered."""

        res = []

        # Create dictionary to process .qasm files more efficiently, each key refers to one benchmark algorithm
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
                    qiskit_gates = utils.timeout_watcher(
                        qiskit_plugin.get_qiskit_gates, [qc], timeout
                    )
                    if not qiskit_gates:
                        break

                    qc_tket = qiskit_to_tk(qc)
                    tket_gates = utils.timeout_watcher(
                        pytket_plugin.get_tket_gates, [qc_tket], timeout
                    )
                    if not tket_gates:
                        break

                    ops_list = qc.count_ops()
                    feature_vector = utils.dict_to_featurevector(
                        ops_list, actual_num_qubits
                    )
                    res.append(
                        (
                            benchmark,
                            feature_vector,
                            qiskit_gates + tket_gates,
                        )
                    )
                except Exception as e:
                    print("fail: ", e)

        jsonString = json.dumps(res, indent=4, sort_keys=True)
        with open("json_data.json", "w") as outfile:
            outfile.write(jsonString)
        return

    def generate_trainingdata_from_json(json_path: str = "json_data.json"):
        """Generates the training data based on the provided json file and defined evaluation score."""

        with open(json_path, "r") as f:
            data = json.load(f)
        training_data = []
        name_list = []
        scores_list = []

        for benchmark in data:
            scores = []
            num_qubits = benchmark[1]["num_qubits"]
            # Qiskit Scores
            if num_qubits > 127:
                continue
            for elem in benchmark[2][1]:
                if elem[0] is None:
                    score = utils.get_width_penalty()
                else:
                    score = utils.calc_score_from_gates_list(
                        elem[0], utils.get_backend_information(elem[1]), num_qubits
                    )

                scores.append(score)
            # Tket Scores
            for elem in benchmark[2][3]:
                if elem[0] is None:
                    score = utils.get_width_penalty()
                else:
                    score = utils.calc_score_from_gates_list(
                        elem[0], utils.get_backend_information(elem[1]), num_qubits
                    )

                scores.append(score)

            training_data.append((list(benchmark[1].values()), np.argmin(scores)))
            name_list.append(benchmark[0])
            scores_list.append(scores)

        return (training_data, name_list, scores_list)

    # def train_neural_network(X, y, name_list=None, actual_scores_list=None):
    #
    #     X, y, indices = np.array(X), np.array(y), np.array(range(len(y)))
    #     X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
    #         X, y, indices, test_size=0.3, random_state=42
    #     )
    #     indices_train, indices_test
    #
    #     model = Sequential()
    #     model.add(Dense(500, activation="relu", input_dim=len(X[0])))
    #     model.add(Dense(100, activation="relu"))
    #     model.add(Dense(50, activation="relu"))
    #     model.add(Dense(10, activation="softmax"))
    #
    #     # Compile the model
    #     model.compile(
    #         optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    #     )
    #     model.fit(X_train, y_train, epochs=100)
    #
    #     scores = model.evaluate(X_train, y_train, verbose=0)
    #     print(
    #         "Accuracy on training data: {}% \n Error on training data: {}".format(
    #             scores[1], 1 - scores[1]
    #         )
    #     )
    #
    #     scores2 = model.evaluate(X_test, y_test, verbose=0)
    #     print(
    #         "Accuracy on test data: {}% \n Error on test data: {}".format(
    #             scores2[1], 1 - scores2[1]
    #         )
    #     )
    #
    #     pred_test = model.predict(X_test)
    #     print(np.mean(np.argmax(model.predict(X_test), axis=1) == y_test))
    #
    #     y_predicted = pred_test
    #     y_actual = y_test
    #     names_list = [name_list[i] for i in indices_test]
    #     scores_filtered = [actual_scores_list[i] for i in indices_test]
    #
    #     circuit_names = []
    #
    #     all_rows = []
    #     all_rows.append(
    #         [
    #             "Benchmark",
    #             "Best Score",
    #             "MQT Predictor",
    #             "Best Machine",
    #             "MQT Predictor",
    #             "Overhead",
    #         ]
    #     )
    #
    #     plt.figure(figsize=(17, 6))
    #     print("len(y_predicted)", len(y_predicted))
    #     for i in range(len(y_predicted)):
    #         y_predicted_instance = np.argmax(y_predicted[i])
    #         if y_predicted_instance != y_actual[i]:
    #             row = []
    #             tmp_res = scores_filtered[i]
    #             circuit_names.append(names_list[i])
    #             machines = get_machines()
    #
    #             comp_val = tmp_res[y_predicted_instance] / tmp_res[y_actual[i]]
    #             row.append(names_list[i])
    #             row.append(np.round(np.min(tmp_res), 2))
    #             row.append(np.round(tmp_res[y_predicted_instance], 2))
    #             row.append(y_actual[i])
    #             row.append(y_predicted_instance)
    #             row.append(np.round(comp_val - 1.00, 2))
    #             all_rows.append(row)
    #
    #             for j in range(10):
    #                 plt.plot(
    #                     len(circuit_names), tmp_res[j], ".", alpha=0.5, label=machines[j]
    #                 )
    #             plt.plot(
    #                 len(circuit_names),
    #                 tmp_res[y_predicted_instance],
    #                 "ko",
    #                 label="MQTPredictor",
    #             )
    #             plt.xlabel(get_machines())
    #
    #             if machines[np.argmin(tmp_res)] != machines[y_predicted_instance]:
    #                 assert np.argmin(tmp_res) == y_actual[i]
    #                 diff = tmp_res[y_predicted_instance] - tmp_res[np.argmin(tmp_res)]
    #                 print(
    #                     names_list[i],
    #                     " predicted: ",
    #                     y_predicted_instance,
    #                     " should be: ",
    #                     y_actual[i],
    #                     " diff: ",
    #                     diff,
    #                 )
    #
    #     plt.title("Evaluation: Compilation Flow Prediction")
    #     plt.xticks(range(len(circuit_names)), circuit_names, rotation=90)
    #     plt.xlabel("Unseen Benchmarks")
    #     plt.ylabel("Actual Score")
    #     handles, labels = plt.gca().get_legend_handles_labels()
    #     by_label = dict(zip(labels, handles))
    #     plt.legend(by_label.values(), by_label.keys(), loc="upper right")
    #     plt.yscale("log")
    #     plt.tight_layout()
    #     plt.savefig("y_pred_eval")
    #
    #     with open("results.csv", "w", encoding="UTF8", newline="") as f:
    #         writer = csv.writer(f)
    #
    #         for row in all_rows:
    #             writer.writerow(row)
    #
    #     return model

    def train_decision_tree_classifier(
        X, y, name_list=None, actual_scores_list=None, max_depth: int = 5
    ):
        import matplotlib.pyplot as plt
        from predictor.src import utils
        from sklearn.tree import plot_tree
        from sklearn import tree

        X, y, indices = np.array(X), np.array(y), np.array(range(len(y)))
        (
            X_train,
            X_test,
            y_train,
            y_test,
            indices_train,
            indices_test,
        ) = train_test_split(X, y, indices, test_size=0.3, random_state=42)

        Predictor._clf = tree.DecisionTreeClassifier(max_depth=max_depth)
        Predictor._clf = Predictor._clf.fit(X_train, y_train)

        y_pred = Predictor._clf.predict(X_test)
        print(np.mean(y_pred == y_test))
        print("Compilation paths from Test Data: ", set(y_test))
        print("Compilation paths from Predictions: ", set(y_pred))
        available_machines = [
            utils.get_machines()[i] for i in set(Predictor._clf.classes_)
        ]

        openqasm_gates_list = utils.get_openqasm_gates()
        res = [openqasm_gates_list[i] for i in range(0, len(openqasm_gates_list))]
        res.append("num_qubits")

        features = np.sort(np.array(res))

        # fig = plt.figure(figsize=(10, 5))
        # plot_tree(
        #     Predictor._clf,
        #     feature_names=features,
        #     class_names=available_machines,
        #     filled=True,
        #     impurity=True,
        #     rounded=True,
        # )
        #
        # plt.savefig("DecisionTreeClassifier.png", dpi=600)
        viz = dtreeviz(
            Predictor._clf,
            X_train,
            y_train,
            target_name="Compilation Path",
            feature_names=features,
            class_names=[i for i in range(9)],
            fancy=True,
        )
        viz.save("fancy_tree.svg")

        names_list = [name_list[i] for i in indices_test]
        scores_filtered = [actual_scores_list[i] for i in indices_test]

        plt.figure(figsize=(17, 6))
        print("len(y_predicted)", len(y_pred))

        # Predictor.plot_eval_all_detailed(names_list, scores_filtered, y_pred, y_test)
        Predictor.plot_eval_all_detailed_compact(
            names_list, scores_filtered, y_pred, y_test
        )
        Predictor.plot_eval_histogram(scores_filtered, y_pred, y_test)

        res = precision_recall_fscore_support(y_test, y_pred)

        with open("metric_table.csv", "w") as csvfile:
            np.savetxt(
                csvfile,
                np.array([list(set(list(y_test) + list(y_pred)))]),
                delimiter=",",
                fmt="%s",
            )
            np.savetxt(csvfile, np.round(np.array(res), 3), delimiter=",", fmt="%s")

        return Predictor._clf

    def plot_eval_histogram(scores_filtered, y_pred, y_test):
        res = []
        for i in range(len(y_pred)):
            # if y_pred[i] != y_test[i]:
            predicted_score = scores_filtered[i][y_pred[i]]
            score = list(np.sort(scores_filtered[i])).index(predicted_score)
            res.append(score + 1)

        assert len(res) == len(y_pred)
        # print(res)

        plt.figure(figsize=(10, 5))
        bars = plt.bar(
            [i for i in range(1, max(res) + 1, 1)],
            height=[res.count(i) / len(res) for i in range(1, max(res) + 1, 1)],
            width=1,
        )
        plt.xticks(
            [i for i in range(1, max(res) + 1, 1)],
            [i for i in range(1, max(res) + 1, 1)],
        )
        plt.xlabel("MQT Predictor Ranking")
        plt.title("Histogram of Predicted Results")
        sum = 0
        for bar in bars:
            yval = bar.get_height()
            rounded_val = str(np.round(yval * 100, 2)) + "%"
            sum += np.round(yval * 100, 2)
            plt.text(bar.get_x() + 0.26, yval + 0.005, rounded_val)

        plt.tick_params(left=False, labelleft=False)
        plt.box(False)
        plt.savefig("MQTPredictor_hist")
        plt.show()
        print("sum: ", sum)

    def plot_eval_all_detailed_compact(names_list, scores_filtered, y_pred, y_test):

        # Create list of all qubit numbers and sort them
        names_list_num_qubits = []
        for i in range(len(names_list)):
            names_list_num_qubits.append(
                int(names_list[i].split("_")[-1].split(".")[0])
            )

        # Sort all other list (names, scores and y_pred) accordingly
        qubit_list_sorted, names_list_sorted_accordingly = zip(
            *sorted(zip(names_list_num_qubits, names_list), key=lambda x: x[0])
        )
        qubit_list_sorted, scores_filtered_sorted_accordingly = zip(
            *sorted(zip(names_list_num_qubits, scores_filtered), key=lambda x: x[0])
        )
        qubit_list_sorted, y_pred_sorted_accordingly = zip(
            *sorted(zip(names_list_num_qubits, y_pred), key=lambda x: x[0])
        )

        for i in range(len(names_list_num_qubits)):
            tmp_res = scores_filtered_sorted_accordingly[i]
            for j in range(len(tmp_res)):
                plt.plot(i, tmp_res[j], "b.", alpha=0.2)

            if y_pred_sorted_accordingly[i] != np.argmin(tmp_res):
                plt.plot(
                    i,
                    tmp_res[y_pred_sorted_accordingly[i]],
                    ".k",
                    label="MQTPredictor non-optimal",
                )
            else:
                plt.plot(
                    i,
                    tmp_res[y_pred_sorted_accordingly[i]],
                    "#ff8600",
                    marker=".",
                    linestyle="None",
                    label="MQTPredictor optimal",
                )

        plt.title("Evaluation: Compilation Flow Prediction")
        plt.xticks(
            [i for i in range(0, len(scores_filtered), 10)],
            [qubit_list_sorted[i] for i in range(0, len(scores_filtered), 10)],
            rotation=90,
        )
        # plt.xticks(range(len(names_list_sorted_accordingly)), names_list_sorted_accordingly, rotation=90)
        plt.xlabel("Benchmark Width (Number of Qubits)")
        plt.ylabel("Actual Score")
        plt.tight_layout()
        y_max = np.sort(np.array(list(set(np.array(scores_filtered).flatten()))))[-2]
        plt.ylim(0, y_max * 1.1)
        plt.xlim(-1, len(scores_filtered) + 1)

        # add vertical lines to annotate the number of possible compilation paths
        x_index = np.where(np.array(qubit_list_sorted) > 8)[0][0]
        plt.axvline(
            x_index, ls="--", color="k", label="# of possible Comp. Paths", linewidth=3
        )
        plt.annotate("10", (x_index - 5, 1))
        x_index = np.where(np.array(qubit_list_sorted) > 11)[0][0]
        plt.axvline(
            x_index, ls="--", color="k", label="# of possible Comp. Paths", linewidth=3
        )
        plt.annotate("8", (x_index - 5, 1))
        x_index = np.where(np.array(qubit_list_sorted) > 27)[0][0]
        plt.axvline(
            x_index, ls="--", color="k", label="# of possible Comp. Paths", linewidth=3
        )
        plt.annotate("6", (x_index - 5, 1))
        x_index = np.where(np.array(qubit_list_sorted) > 80)[0][0]
        plt.axvline(
            x_index, ls="--", color="k", label="# of possible Comp. Paths", linewidth=3
        )
        plt.annotate("4", (x_index - 5, 1))
        x_index = len(scores_filtered)
        plt.axvline(
            x_index, ls="--", color="k", label="# of possible Comp. Paths", linewidth=3
        )
        plt.annotate("2", (x_index - 5, 1))

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc="right")

        plt.savefig("y_pred_eval")

        return

        def plot_eval_all_detailed(names_list, scores_filtered, y_pred, y_test):

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

            for i in range(len(y_pred)):
                # if y_pred[i] != y_test[i]:
                row = []
                tmp_res = scores_filtered[i]
                assert len(tmp_res) == 5 or len(tmp_res) == 10
                circuit_names.append(names_list[i])
                machines = utils.get_machines()

                comp_val = tmp_res[y_pred[i]] / tmp_res[y_test[i]]
                row.append(names_list[i])
                row.append(np.round(np.min(tmp_res), 2))
                row.append(np.round(tmp_res[y_pred[i]], 2))
                row.append(y_test[i])
                row.append(y_pred[i])
                row.append(np.round(comp_val - 1.00, 2))
                all_rows.append(row)

                for j in range(10):
                    plt.plot(
                        len(circuit_names),
                        tmp_res[j],
                        ".",
                        alpha=0.5,
                        label=machines[j],
                    )
                plt.plot(
                    len(circuit_names),
                    tmp_res[y_pred[i]],
                    "ko",
                    label="MQTPredictor",
                )
                plt.xlabel(utils.get_machines())

                if machines[np.argmin(tmp_res)] != machines[y_pred[i]]:
                    assert np.argmin(tmp_res) == y_test[i]
                    diff = tmp_res[y_pred[i]] - tmp_res[np.argmin(tmp_res)]
                    print(
                        names_list[i],
                        " predicted: ",
                        y_pred[i],
                        " should be: ",
                        y_test[i],
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

    def predict(qasm_str_or_path: str):
        """Compilation path prediction for a given qasm string or file path to a qasm file."""
        if ".qasm" in qasm_str_or_path and ".qasm" in qasm_str_or_path:
            print("Reading from .qasm path: ", qasm_str_or_path)
            qc = QuantumCircuit.from_qasm_file(qasm_str_or_path)
        elif QuantumCircuit.from_qasm_str(qasm_str_or_path):
            print("Reading from .qasm str")
            qc = QuantumCircuit.from_qasm_str(qasm_str_or_path)
        else:
            print("Input is neither a .qasm str nor a path to a .qasm file.")
            return

        ops_list = qc.count_ops()
        feature_vector = list(
            utils.dict_to_featurevector(ops_list, qc.num_qubits).values()
        )

        if not (Predictor._clf):
            print("Decision Tree Classifier must be trained first!")
            print(Predictor._clf)
            return

        return Predictor._clf.predict([feature_vector])[0]

    def compile_predicted_compilation_path(qasm_str_or_path: str, prediction: int):
        """Returns the compiled quantum circuit as a qasm string when the original qasm circuit is provided as either
        a string or a file path and the prediction index is given."""
        compilation_path = utils.get_machines()[prediction]

        if ".qasm" in qasm_str_or_path and ".qasm" in qasm_str_or_path:
            print("Reading from .qasm path: ", qasm_str_or_path)
            qc = QuantumCircuit.from_qasm_file(qasm_str_or_path)
        elif QuantumCircuit.from_qasm_str(qasm_str_or_path):
            print("Reading from .qasm str")
            qc = QuantumCircuit.from_qasm_str(qasm_str_or_path)
        qc_tket = qiskit_to_tk(qc)

        if compilation_path == "qiskit_ibm_washington":
            compiled_qc = qiskit_plugin.get_ibm_washington_gates(
                qc, return_circuit=True
            )
        elif compilation_path == "qiskit_ibm_montreal":
            compiled_qc = qiskit_plugin.get_ibm_montreal_gates(qc, return_circuit=True)
        elif compilation_path == "qiskit_ionq":
            compiled_qc = qiskit_plugin.get_ionq_gates(qc, return_circuit=True)
        elif compilation_path == "qiskit_rigetti":
            compiled_qc = qiskit_plugin.get_rigetti_gates(qc, return_circuit=True)
        elif compilation_path == "qiskit_oqc":
            compiled_qc = qiskit_plugin.get_oqc_gates(qc, return_circuit=True)
        elif compilation_path == "tket_ibm_washington":
            compiled_qc = pytket_plugin.get_ibm_washington_gates(
                qc_tket, return_circuit=True
            )
        elif compilation_path == "tket_ibm_montreal":
            compiled_qc = pytket_plugin.get_ibm_montreal_gates(
                qc_tket, return_circuit=True
            )
        elif compilation_path == "tket_ionq":
            compiled_qc = pytket_plugin.get_ionq_gates(qc_tket, return_circuit=True)
        elif compilation_path == "tket_rigetti":
            compiled_qc = pytket_plugin.get_rigetti_gates(qc_tket, return_circuit=True)
        elif compilation_path == "tket_oqc":
            compiled_qc = pytket_plugin.get_oqc_gates(qc_tket, return_circuit=True)
        else:
            print("Compilation Path not found")
            return

        return compiled_qc


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

    Predictor.create_gate_lists_from_folder(timeout=args.timeout)
