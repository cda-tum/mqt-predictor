from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pytket import Circuit

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed, load
from mqt.bench.utils import qiskit_helper, tket_helper
from mqt.predictor import ml, reward, utils
from mqt.predictor.ml import QiskitOptions, TketOptions, TrainingSample
from pytket.extensions.qiskit import tk_to_qiskit
from qiskit import QuantumCircuit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

plt.rcParams["font.family"] = "Times New Roman"

logger = logging.getLogger("mqtpredictor")


class Predictor:
    def __init__(self, verbose: int = 0) -> None:
        if verbose == 1:
            lvl = logging.INFO
        elif verbose == 2:  # noqa: PLR2004
            lvl = logging.DEBUG
        else:
            lvl = logging.WARNING
        logger.setLevel(lvl)

        self.clf = None

    def set_classifier(self, clf: RandomForestClassifier) -> None:
        self.clf = clf

    @staticmethod
    def compile_all_circuits_for_qc(
        filename: str,
        source_path: Path | None = None,
        target_path: Path | None = None,
        timeout: int = 10,
        logger_level: int = logging.INFO,
    ) -> bool:
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

        logger.setLevel(logger_level)
        if source_path is None:
            source_path = ml.helper.get_path_training_circuits()

        if target_path is None:
            target_path = ml.helper.get_path_training_circuits_compiled()

        logger.info("Processing: " + filename)
        qc = QuantumCircuit.from_qasm_file(str(source_path / filename))

        if not qc:
            return False

        compilation_pipeline = ml.helper.get_compilation_pipeline()

        results = []
        comp_path_id = 0
        try:
            for provider_name, devices in compilation_pipeline["devices"].items():
                for device in devices:
                    if device.name == "aria":  # todo: temporary workaround
                        continue

                    for configuration in compilation_pipeline["compiler"]:
                        for compiler, settings in configuration.items():
                            target_filename = filename.split(".qasm")[0] + "_" + str(comp_path_id)
                            comp_path_id += 1
                            if compiler == "qiskit":
                                if qc.num_qubits <= device.num_qubits:
                                    tmp = utils.timeout_watcher(
                                        qiskit_helper.get_mapped_level,
                                        [
                                            qc,
                                            provider_name,
                                            qc.num_qubits,
                                            ml.helper.BackendMapping[device.name],  # todo: temporary fix
                                            cast(QiskitOptions, settings)["optimization_level"],
                                            False,
                                            False,
                                            target_path,
                                            target_filename,
                                        ],
                                        timeout,
                                    )
                                    results.append(tmp)
                            elif compiler == "tket" and qc.num_qubits <= device.num_qubits:
                                tmp = utils.timeout_watcher(
                                    tket_helper.get_mapped_level,
                                    [
                                        qc,
                                        provider_name,
                                        qc.num_qubits,
                                        ml.helper.BackendMapping[device.name],  # todo: temporary fix
                                        cast(TketOptions, settings)["line_placement"],
                                        False,
                                        False,
                                        target_path,
                                        target_filename,
                                    ],
                                    timeout,
                                )
                                results.append(tmp)

            if all(x is False for x in results):
                logger.debug("No compilation succeeded for this quantum circuit: " + filename)
                return False
            return True

        except Exception as e:
            raise RuntimeError("Error during compilation: " + str(e)) from e

    def generate_compiled_circuits(
        self,
        source_path: Path | None = None,
        target_path: Path | None = None,
        timeout: int = 10,
    ) -> None:
        """Handles the creation of all training samples.

        Keyword arguments:
        source_path -- path to file
        target_directory -- path to directory for compiled circuit
        timeout -- timeout in seconds

        """
        if source_path is None:
            source_path = ml.helper.get_path_training_circuits()

        if target_path is None:
            target_path = ml.helper.get_path_training_circuits_compiled()

        path_zip = source_path / "mqtbench_training_samples.zip"
        if not any(file.suffix == ".qasm" for file in source_path.iterdir()) and path_zip.exists():
            import zipfile

            with zipfile.ZipFile(path_zip, "r") as zip_ref:
                zip_ref.extractall(source_path)

        target_path.mkdir(exist_ok=True)

        source_circuits_list = [file.name for file in source_path.iterdir() if file.suffix == ".qasm"]

        Parallel(n_jobs=-1, verbose=100)(
            delayed(self.compile_all_circuits_for_qc)(filename, source_path, target_path, timeout, logger.level)
            for filename in source_circuits_list
        )

    def generate_trainingdata_from_qasm_files(
        self,
        path_uncompiled_circuits: Path | None = None,
        path_compiled_circuits: Path | None = None,
    ) -> tuple[list[TrainingSample], list[str], list[list[float]]]:
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
            path_uncompiled_circuits = ml.helper.get_path_training_circuits()

        if path_compiled_circuits is None:
            path_compiled_circuits = ml.helper.get_path_training_circuits_compiled()

        # init resulting list (feature vector, name, scores)
        training_data = []
        name_list: list[str] = []
        scores_list: list[list[float]] = []

        results: list[tuple[TrainingSample, str, list[float]] | Literal[False]] = Parallel(n_jobs=-1, verbose=100)(
            delayed(self.generate_training_sample)(
                filename.name,
                path_uncompiled_circuits,
                path_compiled_circuits,
                logger.level,
            )
            for filename in path_uncompiled_circuits.iterdir()
        )
        for sample in results:
            if not sample:
                continue

            training_sample, circuit_name, scores = sample
            training_data.append(training_sample)
            name_list.append(circuit_name)
            scores_list.append(scores)

        return training_data, name_list, scores_list

    @staticmethod
    def generate_training_sample(
        file: str,
        path_uncompiled_circuit: Path | None = None,
        path_compiled_circuits: Path | None = None,
        logger_level: int = logging.WARNING,
    ) -> tuple[TrainingSample, str, list[float]] | Literal[False]:

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
        logger.setLevel(logger_level)
        if path_uncompiled_circuit is None:
            path_uncompiled_circuit = ml.helper.get_path_training_circuits()

        if path_compiled_circuits is None:
            path_compiled_circuits = ml.helper.get_path_training_circuits_compiled()

        if ".qasm" not in file:
            msg = f"File {file} is not a qasm file"
            logger.error(msg)
            return False

        compilation_path_dict = ml.helper.get_index_to_compilation_path_dict()
        logger.debug("Checking " + file)
        scores: list[float] = [ml.helper.get_width_penalty()] * len(compilation_path_dict)

        all_relevant_paths = path_compiled_circuits / (file.split(".")[0] + "*")
        all_relevant_files = all_relevant_paths.glob("*")

        for filename in all_relevant_files:
            if (file.split(".")[0] + "_") in str(filename) and filename.suffix == ".qasm":
                comp_path_index = int(str(filename).split("_")[-1].split(".")[0])
                device = compilation_path_dict[comp_path_index]["device"]
                score = reward.expected_fidelity(filename, device)
                scores[comp_path_index] = score

        # raise an error if all scores are the width penalty
        if all(score == ml.helper.get_width_penalty() for score in scores):
            msg = f"File {file} has no valid compilation results"
            logger.error(msg)
            return False

        feature_vec = ml.helper.create_feature_dict(path_uncompiled_circuit / file)
        training_sample = TrainingSample(features=feature_vec, score=float(np.argmax(scores)))
        circuit_name = file.split(".")[0]

        return training_sample, circuit_name, scores

    def train_random_forest_classifier(self, visualize_results: bool = False) -> bool:

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

            logger.info("Best Accuracy: " + str(clf.best_score_))
            top3 = (res.count(1) + res.count(2) + res.count(3)) / len(res)
            logger.info("Top 3: " + str(top3))
            logger.info("Feature Importance: " + str(clf.best_estimator_.feature_importances_))
            self.plot_eval_all_detailed_compact_normed(names_filtered, scores_filtered, y_pred, y_test)

        self.set_classifier(clf.best_estimator_)
        ml.helper.save_classifier(clf.best_estimator_)
        logger.info("Random Forest classifier is trained and saved.")

        return self.clf is not None

    @staticmethod
    def get_prepared_training_data(
        save_non_zero_indices: bool = False,
    ) -> tuple[
        NDArray[np.float_],
        NDArray[np.float_],
        NDArray[np.float_],
        NDArray[np.float_],
        NDArray[np.float_],
        NDArray[np.float_],
        list[str],
        list[list[float]],
    ]:
        training_data, names_list, scores_list = ml.helper.load_training_data()
        X = np.array(data.get_feature_vector() for data in training_data)
        y = np.array(data.score for data in training_data)
        indices = np.array(range(len(y)))

        # Store all non zero feature indices
        non_zero_indices = [i for i in range(len(X[0])) if sum(X[:, i]) > 0]
        X = X[:, non_zero_indices]

        if save_non_zero_indices:
            with Path(ml.helper.get_path_trained_model() / "non_zero_indices.pkl").open("wb") as f:
                pickle.dump(non_zero_indices, f)

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

    @staticmethod
    def calc_performance_measures(
        scores_filtered: list[list[float]],
        y_pred: np.ndarray[Any, np.dtype[np.float64]],
        y_test: np.ndarray[Any, np.dtype[np.float64]],
    ) -> tuple[list[int], list[float]]:
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
            tmp_predicted_score = 0 if predicted_score == ml.helper.get_width_penalty() else predicted_score
            relative_scores.append(tmp_predicted_score - np.max(scores_filtered[i]))
            score = list(np.sort(scores_filtered[i])[::-1]).index(predicted_score)
            res.append(score + 1)

        assert len(res) == len(y_pred)

        return res, relative_scores

    @staticmethod
    def plot_eval_histogram(res: list[int], filename: str = "histogram") -> None:
        """Method to generate the histogram of the resulting ranks

        Keyword arguments:
        res -- all ranks as a list
        filename -- name of the file to save the histogram
        """

        plt.figure(figsize=(10, 5))

        num_of_comp_paths = len(ml.helper.get_index_to_compilation_path_dict())
        plt.bar(
            list(range(0, num_of_comp_paths, 1)),
            height=[res.count(i) / len(res) for i in range(1, num_of_comp_paths + 1, 1)],
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

    @staticmethod
    def plot_eval_all_detailed_compact_normed(
        names_list: list[Any],
        scores_filtered: list[Any],
        y_pred: np.ndarray[Any, np.dtype[np.float64]],
        y_test: np.ndarray[Any, np.dtype[np.float64]],
    ) -> None:
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
            names_list_num_qubits.append(int(names_list[i].split("_")[-1].split(".")[0]))

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
        plt.xlabel("Unseen test circuits (sorted along the number of qubits)", fontsize=18)
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

    def predict(self, qc_or_path: QuantumCircuit | Path) -> Any:
        """Returns a compilation option prediction index for a given qasm file path or qasm string."""

        if self.clf is None:
            path = ml.helper.get_path_trained_model() / "trained_clf.joblib"
            if path.is_file():
                self.clf = load(str(path))
            else:
                error_msg = "Classifier is neither trained nor saved."
                raise FileNotFoundError(error_msg)

        feature_dict = ml.helper.create_feature_dict(qc_or_path)
        feature_vector = list(feature_dict.values())

        with Path(ml.helper.get_path_trained_model() / "non_zero_indices.pkl").open("rb") as f:
            non_zero_indices = pickle.load(f)
        feature_vector = [feature_vector[i] for i in non_zero_indices]

        return self.clf.predict([feature_vector])[0]  # type: ignore[attr-defined]

    @staticmethod
    def compile_as_predicted(qc: QuantumCircuit | Path, prediction: int) -> tuple[QuantumCircuit, int]:
        """Returns the compiled quantum circuit when the original qasm circuit is provided as either
        a string or a file path and the prediction index is given."""

        compilation_path_dict = ml.helper.get_index_to_compilation_path_dict()
        if prediction < 0 or prediction >= len(compilation_path_dict):
            error_msg = "Prediction index is out of range."
            raise IndexError(error_msg)
        if not isinstance(qc, QuantumCircuit):
            if qc.exists():
                logger.info("Reading from .qasm path: " + str(qc))
                qc = QuantumCircuit.from_qasm_file(str(qc))
            else:
                error_msg = "Invalid 'qc' parameter value."
                raise ValueError(error_msg)

        prediction_information = compilation_path_dict[prediction]
        provider_name = prediction_information["provider_name"]
        device_name = ml.helper.BackendMapping[prediction_information["device"].name]  # todo: temporary fix
        compiler = prediction_information["compiler"]
        compiler_settings = prediction_information["compiler_options"]

        if compiler == "qiskit":
            qiskit_compiled_qc: QuantumCircuit = qiskit_helper.get_mapped_level(
                qc, provider_name, qc.num_qubits, device_name, *compiler_settings["qiskit"], False, True
            )
            return qiskit_compiled_qc, prediction
        if compiler == "tket":
            tket_compiled_qc: Circuit = tket_helper.get_mapped_level(
                qc,
                provider_name,
                qc.num_qubits,
                device_name,
                *compiler_settings["tket"],
                False,
                True,
            )
            return (
                tk_to_qiskit(tket_compiled_qc),
                prediction,
            )
        error_msg = "Invalid compiler name."
        raise ValueError(error_msg)

    def instantiate_supervised_ml_model(self, timeout: int) -> None:
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
