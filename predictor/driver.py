import copy

from predictor.src import qiskit_plugin, pytket_plugin, utils

from qiskit import QuantumCircuit
from pytket.extensions.qiskit import qiskit_to_tk

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.tree import plot_tree
from sklearn import tree

from natsort import natsorted
from dtreeviz.trees import dtreeviz


class Predictor:
    _clf = None

    def save_all_compilation_path_results(
        folder_path: str = "./qasm_files",
        timeout: int = 10,
    ):
        """Method to create pre-process data to accelerate the training data generation afterwards. All .qasm files from
        the folder path are considered."""

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
                qc_check = copy.deepcopy(qc)

                print(benchmark)
                if not qc:
                    continue
                actual_num_qubits = qc.num_qubits
                if actual_num_qubits > 127:
                    break
                try:
                    qiskit_opt2 = qiskit_plugin.save_qiskit_compiled_circuits(
                        qc, 2, timeout=timeout, benchmark_name=benchmark
                    )

                    qiskit_opt3 = qiskit_plugin.save_qiskit_compiled_circuits(
                        qc, 3, timeout=timeout, benchmark_name=benchmark
                    )
                    assert qc == qc_check

                    qc_tket = qiskit_to_tk(qc)
                    qc_tket_check = copy.deepcopy(qc_tket)

                    tket_line_True = pytket_plugin.save_tket_compiled_circuits(
                        qc_tket, True, timeout=timeout, benchmark_name=benchmark
                    )

                    tket_line_False = pytket_plugin.save_tket_compiled_circuits(
                        qc_tket, False, timeout=timeout, benchmark_name=benchmark
                    )
                    all_results = (
                        qiskit_opt2 + qiskit_opt3 + tket_line_False + tket_line_True
                    )
                    if all(x is None for x in all_results):
                        break
                    assert qc_tket == qc_tket_check

                except Exception as e:
                    print("fail: ", e)

        return

    def generate_trainingdata_from_qasm_files(
        folder_path: str = "./qasm_files", compiled_path: str = "qasm_compiled/"
    ):
        # init resulting list (feature vector, name, scores)
        training_data = []
        name_list = []
        scores_list = []

        dictionary = {}
        # for each circuit in qasm_files
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
                print("Find: ", benchmark)
                scores = []
                for _ in range(20):
                    scores.append([])
                # iterate over all respective circuits in compiled_path folder
                for filename in os.listdir(compiled_path):
                    # print("Check: ",filename)
                    if benchmark.split(".")[0] in filename and filename.endswith(
                        ".qasm"
                    ):
                        # print("Found: ", filename)
                        # execute function call calc_eval_score_for_qc_and_backend
                        score = utils.calc_eval_score_for_qc(
                            os.path.join(compiled_path, filename)
                        )
                        comp_path_index = int(filename.split("_")[-1].split(".")[0])
                        # print("Comp path index: ", comp_path_index, "\n")
                        scores[comp_path_index] = score

                for i in range(20):
                    if not scores[i]:
                        scores[i] = utils.get_width_penalty()

                feature_vec = utils.create_feature_vector(
                    os.path.join(folder_path, benchmark)
                )
                training_data.append((feature_vec, np.argmin(scores)))
                name_list.append(benchmark.split(".")[0])
                scores_list.append(scores)

        return (training_data, name_list, scores_list)

    def generate_trainingdata_from_json(json_path: str = "json_data.json"):
        """Generates the training data based on the provided json file and defined evaluation score."""

        with open(json_path, "r") as f:
            data = json.load(f)
        training_data = []
        name_list = []
        scores_list = []

        # print(data)
        for benchmark in data:
            scores = []
            num_qubits = benchmark[1]["num_qubits"]
            # Qiskit Scores opt2
            if num_qubits > 127:
                continue

            for elem in benchmark[2][1]:
                if elem[0] is None:
                    score = utils.get_width_penalty()
                else:
                    score = utils.calc_score_from_qc_list(
                        elem[0], utils.get_backend_information(elem[1]), num_qubits
                    )

                scores.append(score)
            assert len(scores) == 5
            # Qiskit Scores opt3
            if num_qubits > 127:
                continue
            for elem in benchmark[2][3]:
                if elem[0] is None:
                    score = utils.get_width_penalty()
                else:
                    score = utils.calc_score_from_qc_list(
                        elem[0], utils.get_backend_information(elem[1]), num_qubits
                    )

                scores.append(score)
            assert len(scores) == 10

            # Tket Scores Lineplacement

            for elem in benchmark[2][5]:
                if elem[0] is None:
                    score = utils.get_width_penalty()
                else:
                    score = utils.calc_score_from_qc_list(
                        elem[0], utils.get_backend_information(elem[1]), num_qubits
                    )

                scores.append(score)
            assert len(scores) == 15

            # Tket Scores Graphplacement

            for elem in benchmark[2][7]:
                if elem[0] is None:
                    score = utils.get_width_penalty()
                else:
                    score = utils.calc_score_from_qc_list(
                        elem[0], utils.get_backend_information(elem[1]), num_qubits
                    )

                scores.append(score)
            assert len(scores) == 19

            training_data.append((list(benchmark[1].values()), np.argmin(scores)))
            name_list.append(benchmark[0])
            scores_list.append(scores)

        return (training_data, name_list, scores_list)

    def train_decision_tree_classifier(
        X, y, name_list=None, actual_scores_list=None, max_depth: int = 5
    ):

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
        print("Compilation paths from Train Data: ", set(y_train))
        print("Compilation paths from Test Data: ", set(y_test))
        print("Compilation paths from Predictions: ", set(y_pred))
        available_machines = [
            utils.get_machines()[i] for i in set(Predictor._clf.classes_)
        ]

        openqasm_qc_list = utils.get_openqasm_qc()
        res = [openqasm_qc_list[i] for i in range(0, len(openqasm_qc_list))]
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
            class_names=list(Predictor._clf.classes_),
            fancy=True,
        )
        viz.save("fancy_tree.svg")

        names_list = [name_list[i] for i in indices_test]
        scores_filtered = [actual_scores_list[i] for i in indices_test]

        # print("len(y_predicted)", len(y_pred))

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

        return np.mean(y_pred == y_test)

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
        plt.figure(figsize=(17, 6))
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
        )
        # plt.xticks(range(len(names_list_sorted_accordingly)), names_list_sorted_accordingly, rotation=90)
        plt.xlabel("Benchmark Width (Number of Qubits)")
        plt.ylabel("Actual Score")
        plt.tight_layout()
        y_max = np.sort(np.array(list(set(np.array(scores_filtered).flatten()))))[-2]
        plt.ylim(0, y_max * 1.1)
        plt.xlim(-1, len(scores_filtered) + 1)

        # add vertical lines to annotate the number of possible compilation paths
        if len(np.where(np.array(qubit_list_sorted) > 11)) > 1:
            x_index = np.where(np.array(qubit_list_sorted) > 8)[0][0]
            plt.axvline(
                x_index,
                ls="--",
                color="k",
                label="# of possible Comp. Paths",
                linewidth=3,
            )
            plt.annotate("19", (x_index - 5, 1))

            if len(np.where(np.array(qubit_list_sorted) > 11)) > 1:
                x_index = np.where(np.array(qubit_list_sorted) > 11)[0][0]
                plt.axvline(
                    x_index,
                    ls="--",
                    color="k",
                    label="# of possible Comp. Paths",
                    linewidth=3,
                )
                plt.annotate("16", (x_index - 5, 1))
                if len(np.where(np.array(qubit_list_sorted) > 27)) > 1:
                    x_index = np.where(np.array(qubit_list_sorted) > 27)[0][0]
                    plt.axvline(
                        x_index,
                        ls="--",
                        color="k",
                        label="# of possible Comp. Paths",
                        linewidth=3,
                    )
                    plt.annotate("12", (x_index - 5, 1))
                    if len(np.where(np.array(qubit_list_sorted) > 80)) > 1:
                        x_index = np.where(np.array(qubit_list_sorted) > 80)[0][0]
                        plt.axvline(
                            x_index,
                            ls="--",
                            color="k",
                            label="# of possible Comp. Paths",
                            linewidth=3,
                        )
                        plt.annotate("8", (x_index - 5, 1))
                        x_index = len(scores_filtered)
                        plt.axvline(
                            x_index,
                            ls="--",
                            color="k",
                            label="# of possible Comp. Paths",
                            linewidth=3,
                        )
                        plt.annotate("4", (x_index - 5, 1))

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc="right")

        plt.savefig("y_pred_eval")

        return

    def predict(qasm_path: str):
        """Compilation path prediction for a given qasm file path to a qasm file."""
        if not(".qasm" in qasm_path and ".qasm" in qasm_path):
            print("Input is neither a .qasm str nor a path to a .qasm file.")
            return

        feature_vector = list(utils.create_feature_vector(qasm_path))

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

        if compilation_path == "qiskit_ionq_opt2":
            compiled_qc = qiskit_plugin.get_ionq_qc(qc, 2)
        elif compilation_path == "qiskit_ibm_washington_opt2":
            compiled_qc = qiskit_plugin.get_ibm_washington_qc(qc, 2)
        elif compilation_path == "qiskit_ibm_montreal_opt2":
            compiled_qc = qiskit_plugin.get_ibm_montreal_qc(qc, 2)
        elif compilation_path == "qiskit_rigetti_opt2":
            compiled_qc = qiskit_plugin.get_rigetti_qc(qc, 2)
        elif compilation_path == "qiskit_oqc_opt2":
            compiled_qc = qiskit_plugin.get_oqc_qc(qc, 2)
        elif compilation_path == "qiskit_ionq_opt3":
            compiled_qc = qiskit_plugin.get_ionq_qc(qc, 3)
        elif compilation_path == "qiskit_ibm_washington_opt3":
            compiled_qc = qiskit_plugin.get_ibm_washington_qc(qc, 3)
        elif compilation_path == "qiskit_ibm_montreal_opt3":
            compiled_qc = qiskit_plugin.get_ibm_montreal_qc(qc, 3)
        elif compilation_path == "qiskit_rigetti_opt3":
            compiled_qc = qiskit_plugin.get_rigetti_qc(qc, 3)
        elif compilation_path == "qiskit_oqc_opt3":
            compiled_qc = qiskit_plugin.get_oqc_qc(qc, 3)
        elif compilation_path == "tket_ionq":
            compiled_qc = pytket_plugin.get_ionq_qc(qc_tket)
        elif compilation_path == "tket_ibm_washington_line":
            compiled_qc = pytket_plugin.get_ibm_washington_qc(qc_tket, True)
        elif compilation_path == "tket_ibm_montreal_line":
            compiled_qc = pytket_plugin.get_ibm_montreal_qc(qc_tket, True)
        elif compilation_path == "tket_rigetti_line":
            compiled_qc = pytket_plugin.get_rigetti_qc(qc_tket, True)
        elif compilation_path == "tket_oqc_line":
            compiled_qc = pytket_plugin.get_oqc_qc(qc_tket, True)
        elif compilation_path == "tket_ibm_washington_graph":
            compiled_qc = pytket_plugin.get_ibm_washington_qc(qc_tket, False)
        elif compilation_path == "tket_ibm_montreal_graph":
            compiled_qc = pytket_plugin.get_ibm_montreal_qc(qc_tket, False)
        elif compilation_path == "tket_rigetti_graph":
            compiled_qc = pytket_plugin.get_rigetti_qc(qc_tket, False)
        elif compilation_path == "tket_oqc_graph":
            compiled_qc = pytket_plugin.get_oqc_qc(qc_tket, False)
        else:
            print("Compilation Path not found")
            return

        return compiled_qc


if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description="Create Training Data")
    #
    # parser.add_argument("--timeout", type=int, default=10)
    # parser.add_argument("--path", type=str, default="test/")
    #
    # args = parser.parse_args()
    #
    # Predictor.save_all_compilation_path_results(
    #     folder_path=args.path, timeout=args.timeout
    # )
    Predictor.generate_trainingdata_from_qasm_files(folder_path="parsetest/")
