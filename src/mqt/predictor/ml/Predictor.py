import glob
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed, load
from mqt.bench.utils import qiskit_helper, tket_helper
from pytket.extensions.qiskit import tk_to_qiskit
from qiskit import QuantumCircuit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

from mqt.predictor import ml, reward, utils

plt.rcParams["font.family"] = "Times New Roman"


class Predictor:
    def __init__(self, verbose=0):
        self.verbose = verbose

        self.logger = logging.getLogger("mqtpredictor")
        if verbose == 1:
            self.logger.setLevel(logging.INFO)
        elif verbose == 2:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.WARNING)

        self.clf = None

    def set_classifier(self, clf):
        self.clf = clf

    def compile_all_circuits_for_qc(
        self,
        filename: str,
        source_path: str = None,
        target_path: str = None,
        timeout: int = 10,
    ):
        """Handles the creation of one training sample.

        Keyword arguments:
        filename -- qasm circuit sample filename
        source_path -- path to file
        target_path -- path to directory for compiled circuit
        timeout -- timeout in seconds

        Return values:
        True -- at least one compilation option succeeded
        False -- if not
        """
        if source_path is None:
            source_path = str(ml.helper.get_path_training_circuits())

        if target_path is None:
            target_path = str(ml.helper.get_path_training_circuits_compiled())

        self.logger.info("Processing: " + filename)
        qc = QuantumCircuit.from_qasm_file(Path(source_path) / filename)

        if not qc:
            return False

        compilation_pipeline = ml.helper.get_compilation_pipeline()

        results = []
        comp_path_id = 0
        try:
            for gate_set_name, devices in compilation_pipeline.get("devices").items():
                for device_name, max_qubits in devices:
                    for compiler, settings in compilation_pipeline["compiler"].items():
                        if "qiskit" in compiler:
                            for opt_level in settings["optimization_level"]:
                                target_filename = (
                                    filename.split(".qasm")[0] + "_" + str(comp_path_id)
                                )
                                comp_path_id += 1
                                if max_qubits >= qc.num_qubits:
                                    tmp = utils.timeout_watcher(
                                        qiskit_helper.get_mapped_level,
                                        [
                                            qc,
                                            gate_set_name,
                                            qc.num_qubits,
                                            device_name,
                                            opt_level,
                                            False,
                                            False,
                                            target_path,
                                            target_filename,
                                        ],
                                        timeout,
                                    )
                                    results.append(tmp)
                                    if not tmp:
                                        continue
                        elif "tket" in compiler:
                            for lineplacement in settings["lineplacement"]:
                                target_filename = (
                                    filename.split(".qasm")[0] + "_" + str(comp_path_id)
                                )
                                comp_path_id += 1
                                if max_qubits >= qc.num_qubits:
                                    tmp = utils.timeout_watcher(
                                        tket_helper.get_mapped_level,
                                        [
                                            qc,
                                            gate_set_name,
                                            qc.num_qubits,
                                            device_name,
                                            lineplacement,
                                            False,
                                            False,
                                            target_path,
                                            target_filename,
                                        ],
                                        timeout,
                                    )
                                    results.append(tmp)
                                    if not tmp:
                                        continue

            if all(x is False for x in results):
                self.logger.debug(
                    "No compilation succeeded for this quantum circuit: " + filename
                )
                return False
            return True

        except Exception as e:
            raise RuntimeError("Error during compilation: " + str(e)) from e

    def generate_compiled_circuits(
        self,
        source_path: str = None,
        target_path: str = None,
        timeout: int = 10,
    ):
        """Handles the creation of all training samples.

        Keyword arguments:
        source_path -- path to file
        target_directory -- path to directory for compiled circuit
        timeout -- timeout in seconds

        """
        if source_path is None:
            source_path = str(ml.helper.get_path_training_circuits())

        if target_path is None:
            target_path = str(ml.helper.get_path_training_circuits_compiled())

        path_zip = Path(source_path) / "mqtbench_training_samples.zip"
        if (
            not any(file.suffix == ".qasm" for file in Path(source_path).iterdir())
            and path_zip.exists()
        ):
            path_zip = str(path_zip)
            import zipfile

            with zipfile.ZipFile(path_zip, "r") as zip_ref:
                zip_ref.extractall(source_path)

        Path(target_path).mkdir(exist_ok=True)

        source_circuits_list = [
            file.name for file in Path(source_path).iterdir() if file.suffix == ".qasm"
        ]

        Parallel(n_jobs=-1, verbose=100)(
            delayed(self.compile_all_circuits_for_qc)(
                filename, source_path, target_path, timeout
            )
            for filename in source_circuits_list
        )

    def generate_trainingdata_from_qasm_files(
        self,
        path_uncompiled_circuits: str = None,
        path_compiled_circuits: str = None,
    ):
        """Handles to create training data from all generated training samples

        Keyword arguments:
        path_uncompiled_circuits -- path to file
        path_compiled_circuits -- path to directory for compiled circuit

        Return values:
        training_data_ML_aggregated -- training data
        name_list -- names of all training samples
        scores -- evaluation scores for all compilation options
        """
        if path_uncompiled_circuits is None:
            path_uncompiled_circuits = str(ml.helper.get_path_training_circuits())

        if path_compiled_circuits is None:
            path_compiled_circuits = str(
                ml.helper.get_path_training_circuits_compiled()
            )

        # init resulting list (feature vector, name, scores)
        training_data = []
        name_list = []
        scores_list = []

        results = Parallel(n_jobs=-1, verbose=100)(
            delayed(self.generate_training_sample)(
                str(filename.name), path_uncompiled_circuits, path_compiled_circuits
            )
            for filename in Path(path_uncompiled_circuits).iterdir()
        )
        for sample in results:
            if not sample:
                continue

            training_sample, circuit_name, scores = sample
            training_data.append(training_sample)
            name_list.append(circuit_name)
            scores_list.append(scores)

        return (training_data, name_list, scores_list)

    def generate_training_sample(
        self,
        file: str,
        path_uncompiled_circuit: str = None,
        path_compiled_circuits: str = None,
    ):
        """Handles to create training data from a single generated training sample

        Keyword arguments:
        file -- filename for the training sample
        path_uncompiled_circuit -- path to file
        path_compiled_circuits -- path to directory for compiled circuit

        Return values:
        training_sample -- training data sample
        circuit_name -- names of the training sample circuit
        scores -- evaluation scores for all compilation options
        """

        if path_uncompiled_circuit is None:
            path_uncompiled_circuit = str(ml.helper.get_path_training_circuits())

        if path_compiled_circuits is None:
            path_compiled_circuits = str(
                ml.helper.get_path_training_circuits_compiled()
            )

        if ".qasm" not in file:
            return False

        LUT = ml.helper.get_index_to_comppath_LUT()
        self.logger.debug("Checking " + file)
        scores = []
        for _ in range(len(LUT)):
            scores.append([])
        all_relevant_paths = Path(path_compiled_circuits) / (file.split(".")[0] + "*")
        all_relevant_files = glob.glob(str(all_relevant_paths))

        for filename in all_relevant_files:
            if (file.split(".")[0] + "_") in filename and filename.endswith(".qasm"):
                comp_path_index = int(filename.split("_")[-1].split(".")[0])
                device = LUT.get(comp_path_index)[1]

                score = reward.expected_fidelity(filename, device)
                scores[comp_path_index] = score

        num_not_empty_entries = 0
        for i in range(len(LUT)):
            if not scores[i]:
                scores[i] = ml.helper.get_width_penalty()
            else:
                num_not_empty_entries += 1

        if num_not_empty_entries == 0:
            return False

        feature_vec = ml.helper.create_feature_dict(
            str(Path(path_uncompiled_circuit) / file)
        )
        training_sample = (list(feature_vec.values()), np.argmax(scores))
        circuit_name = file.split(".")[0]

        return (training_sample, circuit_name, scores)

    def train_random_forest_classifier(self, visualize_results=False):

        (
            X_train,
            X_test,
            y_train,
            y_test,
            indices_train,
            indices_test,
            names_list,
            scores_list,
        ) = self.get_prepared_training_data(save_non_zero_indices=True)

        scores_filtered = [scores_list[i] for i in indices_test]
        names_filtered = [names_list[i] for i in indices_test]

        tree_param = [
            {
                "n_estimators": [100, 200, 500],
                "max_depth": list(range(8, 30, 6)),
                "min_samples_split": list(range(2, 20, 6)),
                "min_samples_leaf": list(range(2, 20, 6)),
                "bootstrap": [True, False],
            },
        ]

        clf = RandomForestClassifier(random_state=0)
        clf = GridSearchCV(clf, tree_param, cv=5, n_jobs=8).fit(X_train, y_train)

        if visualize_results:
            y_pred = np.array(list(clf.predict(X_test)))
            res, _ = self.calc_performance_measures(scores_filtered, y_pred, y_test)
            self.plot_eval_histogram(res, filename="RandomForestClassifier")

            self.logger.info("Best Accuracy: " + clf.best_score_)
            top3 = (res.count(1) + res.count(2) + res.count(3)) / len(res)
            self.logger.info("Top 3: " + top3)
            self.logger.info(
                "Feature Importance: " + clf.best_estimator_.feature_importances_
            )

            self.plot_eval_all_detailed_compact_normed(
                names_filtered, scores_filtered, y_pred, y_test
            )

        self.set_classifier(clf.best_estimator_)
        ml.helper.save_classifier(clf.best_estimator_)
        self.logger.info("Random Forest classifier is trained and saved.")

        return self.clf is not None

    def get_prepared_training_data(self, save_non_zero_indices=False):
        training_data, names_list, scores_list = ml.helper.load_training_data()
        X, y = zip(*training_data)
        X = list(X)
        y = list(y)
        for i in range(len(X)):
            X[i] = list(X[i])
            scores_list[i] = list(scores_list[i])

        X, y, indices = np.array(X), np.array(y), np.array(range(len(y)))

        # Store all non zero feature indices
        non_zero_indices = []
        for i in range(len(X[0])):
            if sum(X[:, i]) > 0:
                non_zero_indices.append(i)
        X = X[:, non_zero_indices]

        if save_non_zero_indices:
            data = np.asarray(non_zero_indices)
            np.save(ml.helper.get_path_trained_model() / "non_zero_indices.npy", data)

        (
            X_train,
            X_test,
            y_train,
            y_test,
            indices_train,
            indices_test,
        ) = train_test_split(X, y, indices, test_size=0.3, random_state=5)

        return (
            X_train,
            X_test,
            y_train,
            y_test,
            indices_train,
            indices_test,
            names_list,
            scores_list,
        )

    def calc_performance_measures(self, scores_filtered, y_pred, y_test):
        """Method to generate the performance measures for a trained classifier

        Keyword arguments:
        scores_filtered -- ground truth of all combinations of compilation options
        y_pred -- predicted combination of compilation options
        y_test -- best combination of compilation options

        Return values:
        res -- list of all ranks
        relative_scores -- performance difference to best score

        """

        res = []
        relative_scores = []
        for i in range(len(y_pred)):
            assert np.argmax(scores_filtered[i]) == y_test[i]
            predicted_score = scores_filtered[i][y_pred[i]]
            if predicted_score == ml.helper.get_width_penalty():
                tmp_predicted_score = 0
            else:
                tmp_predicted_score = predicted_score
            relative_scores.append(tmp_predicted_score - np.max(scores_filtered[i]))
            score = list(np.sort(scores_filtered[i])[::-1]).index(predicted_score)
            res.append(score + 1)

        assert len(res) == len(y_pred)

        return res, relative_scores

    def plot_eval_histogram(self, res, filename="histogram"):
        """Method to generate the histogram of the resulting ranks

        Keyword arguments:
        res -- all ranks as a list
        filename -- name of the file to save the histogram
        """

        plt.figure(figsize=(10, 5))

        num_of_comp_paths = len(ml.helper.get_index_to_comppath_LUT())
        plt.bar(
            list(range(0, num_of_comp_paths, 1)),
            height=[
                res.count(i) / len(res) for i in range(1, num_of_comp_paths + 1, 1)
            ],
            width=1,
        )
        plt.xticks(
            list(range(0, num_of_comp_paths, 1)),
            list(range(1, num_of_comp_paths + 1, 1)),
            fontsize=16,
        )
        plt.yticks(fontsize=16)

        plt.xlabel(
            "Best prediction                                                        Worst prediction",
            fontsize=18,
        )
        plt.ylabel("Relative frequency", fontsize=18)
        result_path = Path("results")
        if not result_path.is_dir():
            result_path.mkdir()
        plt.savefig(result_path / (filename + ".pdf"))
        plt.show()

    def plot_eval_all_detailed_compact_normed(
        self, names_list, scores_filtered, y_pred, y_test
    ):
        """Method to generate the detailed graph to examine the differences in evaluation scores

        Keyword arguments:
        names_list -- all names filtered for the respectively predicted indices of all training data
        scores_filtered -- all scores filtered for the respectively predicted indices of all training data
        y_pred -- predicted labels
        y_test -- actual labels
        """

        # Create list of all qubit numbers and sort them
        names_list_num_qubits = []
        for i in range(len(names_list)):
            assert np.argmax(scores_filtered[i]) == y_test[i]
            names_list_num_qubits.append(
                int(names_list[i].split("_")[-1].split(".")[0])
            )

        # Sort all other list (num_qubits, scores and y_pred) accordingly
        (
            qubit_list_sorted,
            scores_filtered_sorted_accordingly,
            y_pred_sorted_accordingly,
        ) = zip(*sorted(zip(names_list_num_qubits, scores_filtered, y_pred)))
        plt.figure(figsize=(17, 8))
        for i in range(len(names_list_num_qubits)):
            tmp_res = scores_filtered_sorted_accordingly[i]
            max_score = max(tmp_res)
            for j in range(len(tmp_res)):
                plt.plot(i, tmp_res[j] / max_score, "b.", alpha=1.0, markersize=1.7)

            plt.plot(
                i,
                tmp_res[y_pred_sorted_accordingly[i]] / max_score,
                "#ff8600",
                marker=".",
                linestyle="None",
            )

        plt.xticks(
            list(range(0, len(scores_filtered), 20)),
            [qubit_list_sorted[i] for i in range(0, len(scores_filtered), 20)],
            fontsize=18,
        )
        plt.yticks(fontsize=18)
        plt.xlabel(
            "Unseen test circuits (sorted along the number of qubits)", fontsize=18
        )
        plt.ylabel(
            "Evaluation scores of combinations of options \n (normalized per test circuit)",
            fontsize=18,
        )
        plt.tight_layout()

        plt.ylim(0, 1.05)
        plt.xlim(0, len(scores_filtered))
        result_path = Path("results")
        if not result_path.is_dir():
            result_path.mkdir()
        plt.savefig(result_path / "y_pred_eval_normed.pdf")

    def predict(self, qasm_str_or_path: str):
        """Returns a compilation option prediction index for a given qasm file path or qasm string."""

        if self.clf is None:
            path = ml.helper.get_path_trained_model() / "trained_clf.joblib"
            if path.is_file():
                self.clf = load(str(path))
            else:
                raise FileNotFoundError("Classifier is neither trained nor saved.")

        feature_dict = ml.helper.create_feature_dict(qasm_str_or_path)
        if not feature_dict:
            return None
        feature_vector = list(feature_dict.values())

        path = ml.helper.get_path_trained_model() / "non_zero_indices.npy"
        non_zero_indices = np.load(str(path), allow_pickle=True)
        feature_vector = [feature_vector[i] for i in non_zero_indices]

        return self.clf.predict([feature_vector])[0]

    def compile_as_predicted(self, qc: str, prediction: int):
        """Returns the compiled quantum circuit when the original qasm circuit is provided as either
        a string or a file path and the prediction index is given."""

        LUT = ml.helper.get_index_to_comppath_LUT()
        if prediction < 0 or prediction >= len(LUT):
            raise IndexError("Prediction index is out of range.")
        if not isinstance(qc, QuantumCircuit):
            if Path(qc).exists():
                self.logger.info("Reading from .qasm path: " + qc)
                qc = QuantumCircuit.from_qasm_file(qc)
            elif QuantumCircuit.from_qasm_str(qc):
                self.logger.info("Reading from .qasm str")
                qc = QuantumCircuit.from_qasm_str(qc)
            else:
                raise ValueError("Invalid 'qc' parameter value.")

        prediction_information = LUT.get(prediction)
        gate_set_name = prediction_information[0]
        device = prediction_information[1]
        compiler = prediction_information[2]
        compiler_settings = prediction_information[3]

        if compiler == "qiskit":
            compiled_qc = qiskit_helper.get_mapped_level(
                qc, gate_set_name, qc.num_qubits, device, compiler_settings, False, True
            )
            return compiled_qc, ml.helper.get_index_to_comppath_LUT()[prediction]
        elif compiler == "tket":
            compiled_qc = tket_helper.get_mapped_level(
                qc,
                gate_set_name,
                qc.num_qubits,
                device,
                compiler_settings,
                False,
                True,
            )
            return (
                tk_to_qiskit(compiled_qc),
                ml.helper.get_index_to_comppath_LUT()[prediction],
            )
        else:
            raise ValueError("Invalid compiler name.")

    def instantiate_supervised_ML_model(self, timeout):
        # Generate compiled circuits and save them as qasm files
        self.generate_compiled_circuits(
            timeout=timeout,
        )
        # Generate training data from qasm files
        res = self.generate_trainingdata_from_qasm_files()
        # Save those training data for faster re-processing
        ml.helper.save_training_data(res)
        # Train the Random Forest Classifier on created training data
        self.train_random_forest_classifier()
