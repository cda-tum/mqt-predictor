from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed, load
from qiskit import QuantumCircuit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

from mqt.predictor import ml, reward, rl, utils

if TYPE_CHECKING:
    from numpy._typing import NDArray

plt.rcParams["font.family"] = "Times New Roman"

logger = logging.getLogger("mqt-predictor")


class Predictor:
    def __init__(self, logger_level: int = logging.INFO) -> None:
        logger.setLevel(logger_level)

        self.clf = None

    def set_classifier(self, clf: RandomForestClassifier) -> None:
        """Sets the classifier to the given classifier"""
        self.clf = clf

    def compile_all_circuits_circuitwise(
        self,
        figure_of_merit: reward.figure_of_merit,
        timeout: int,
        source_path: Path | None = None,
        target_path: Path | None = None,
        logger_level: int = logging.INFO,
    ) -> None:
        """Compiles all circuits in the given directory with the given timeout and saves them in the given directory.

        Args:
            timeout (int): The timeout in seconds for the compilation of a single circuit
            source_path (Path, optional): The path to the directory containing the circuits to be compiled. Defaults to None.
            target_path (Path, optional): The path to the directory where the compiled circuits should be saved. Defaults to None.
            logger_level (int, optional): The level of the logger. Defaults to logging.INFO.

        """
        logger.setLevel(logger_level)

        if source_path is None:
            source_path = ml.helper.get_path_training_circuits()

        if target_path is None:
            target_path = ml.helper.get_path_training_circuits_compiled()

        Parallel(n_jobs=-1, verbose=100)(
            delayed(self.generate_compiled_circuits_for_single_training_circuit)(
                filename, timeout, source_path, target_path, figure_of_merit
            )
            for filename in source_path.iterdir()
        )

    def generate_compiled_circuits_for_single_training_circuit(
        self,
        filename: Path,
        timeout: int,
        source_path: Path,
        target_path: Path,
        figure_of_merit: reward.figure_of_merit,
    ) -> None:
        """Compiles a single circuit with the given timeout and saves it in the given directory.

        Args:
            filename (Path): The path to the circuit to be compiled
            timeout (int): The timeout in seconds for the compilation of the circuit
            source_path (Path): The path to the directory containing the circuit to be compiled
            target_path (Path): The path to the directory where the compiled circuit should be saved
            figure_of_merit (reward.figure_of_merit): The figure of merit to be used for compilation.

        """
        try:
            qc = QuantumCircuit.from_qasm_file(Path(source_path) / filename)
            if filename.suffix != ".qasm":
                return

            for i, dev in enumerate(rl.helper.get_devices()):
                target_filename = str(filename).split("/")[-1].split(".qasm")[0] + "_" + figure_of_merit + "_" + str(i)
                if (Path(target_path) / (target_filename + ".qasm")).exists() or qc.num_qubits > dev["max_qubits"]:
                    continue
                try:
                    res = utils.timeout_watcher(rl.qcompile, [qc, figure_of_merit, dev["name"]], timeout)
                    if res:
                        compiled_qc = res[0]
                        compiled_qc.qasm(filename=Path(target_path) / (target_filename + ".qasm"))

                except Exception as e:
                    print(e, filename, "inner")

        except Exception as e:
            print(e, filename, "outer")

    def compile_all_circuits_devicewise(
        self,
        device_name: str,
        timeout: int,
        figure_of_merit: reward.figure_of_merit,
        source_path: Path | None = None,
        target_path: Path | None = None,
        logger_level: int = logging.INFO,
    ) -> None:
        """Compiles all circuits in the given directory with the given timeout and saves them in the given directory.

        Args:
            device_name (str): The name of the device to be used for compilation
            timeout (int): The timeout in seconds for the compilation of a single circuit
            figure_of_merit (reward.reward_functions): The figure of merit to be used for compilation
            source_path (Path, optional): The path to the directory containing the circuits to be compiled. Defaults to None.
            target_path (Path, optional): The path to the directory where the compiled circuits should be saved. Defaults to None.
            logger_level (int, optional): The level of the logger. Defaults to logging.INFO.
        """
        logger.setLevel(logger_level)

        logger.info("Processing: " + device_name + " for " + figure_of_merit)
        rl_pred = rl.Predictor(figure_of_merit=figure_of_merit, device_name=device_name)

        dev_index = rl.helper.get_device_index_of_device(device_name)
        dev_max_qubits = rl.helper.get_devices()[dev_index]["max_qubits"]

        if source_path is None:
            source_path = ml.helper.get_path_training_circuits()

        if target_path is None:
            target_path = ml.helper.get_path_training_circuits_compiled()

        for filename in source_path.iterdir():
            if filename.suffix != ".qasm":
                continue
            qc = QuantumCircuit.from_qasm_file(Path(source_path) / filename)
            if qc.num_qubits > dev_max_qubits:
                continue

            target_filename = (
                str(filename).split("/")[-1].split(".qasm")[0] + "_" + figure_of_merit + "_" + str(dev_index)
            )
            if (Path(target_path) / (target_filename + ".qasm")).exists():
                continue
            try:
                res = utils.timeout_watcher(rl.qcompile, [qc, figure_of_merit, device_name, rl_pred], timeout)
                if res:
                    compiled_qc = res[0]
                    compiled_qc.qasm(filename=Path(target_path) / (target_filename + ".qasm"))

            except Exception as e:
                print(e, filename, device_name)
                raise RuntimeError("Error during compilation: " + str(e)) from e

    def generate_compiled_circuits(
        self,
        source_path: Path | None = None,
        target_path: Path | None = None,
        timeout: int = 600,
    ) -> None:
        """Compiles all circuits in the given directory with the given timeout and saves them in the given directory.

        Args:
            source_path (Path, optional): The path to the directory containing the circuits to be compiled. Defaults to None.
            target_path (Path, optional): The path to the directory where the compiled circuits should be saved. Defaults to None.
            timeout (int, optional): The timeout in seconds for the compilation of a single circuit. Defaults to 600.
        """
        if source_path is None:
            source_path = ml.helper.get_path_training_circuits()

        if target_path is None:
            target_path = ml.helper.get_path_training_circuits_compiled()

        path_zip = source_path / "training_data_device_selection.zip"
        if not any(file.suffix == ".qasm" for file in source_path.iterdir()) and path_zip.exists():
            import zipfile

            with zipfile.ZipFile(str(path_zip), "r") as zip_ref:
                zip_ref.extractall(source_path)

        target_path.mkdir(exist_ok=True)

        Parallel(n_jobs=1, verbose=100)(
            delayed(self.compile_all_circuits_devicewise)(
                device_name, timeout, figure_of_merit, source_path, target_path, logger.level
            )
            for figure_of_merit in ["expected_fidelity", "critical_depth"]
            for device_name in [dev["name"] for dev in rl.helper.get_devices()]
        )

    def generate_trainingdata_from_qasm_files(
        self,
        figure_of_merit: reward.figure_of_merit,
        path_uncompiled_circuits: Path | None = None,
        path_compiled_circuits: Path | None = None,
    ) -> tuple[list[NDArray[np.float_]], list[str], list[NDArray[np.float_]]]:
        """Handles to create training data from all generated training samples

        Args:
            figure_of_merit (reward.reward_functions): The figure of merit to be used for training
            path_uncompiled_circuits (Path, optional): The path to the directory containing the uncompiled circuits. Defaults to None.
            path_compiled_circuits (Path, optional): The path to the directory containing the compiled circuits. Defaults to None.

        Returns:
            tuple[list[Any], list[Any], list[Any]]: The training data, consisting of training_data, name_list, scores_list

        """
        if not path_uncompiled_circuits:
            path_uncompiled_circuits = ml.helper.get_path_training_circuits()

        if not path_compiled_circuits:
            path_compiled_circuits = ml.helper.get_path_training_circuits_compiled()

        # init resulting list (feature vector, name, scores)
        training_data = []
        name_list = []
        scores_list = []

        results = Parallel(n_jobs=-1, verbose=100)(
            delayed(self.generate_training_sample)(
                filename.name,
                path_uncompiled_circuits,
                path_compiled_circuits,
                figure_of_merit,
                logger.level,
            )
            for filename in path_uncompiled_circuits.glob("*.qasm")
        )
        for sample in results:
            training_sample, circuit_name, scores = sample
            if all(score == -1 for score in scores):
                continue
            training_data.append(training_sample)
            name_list.append(circuit_name)
            scores_list.append(scores)

        return (training_data, name_list, scores_list)

    def generate_training_sample(
        self,
        file: Path,
        path_uncompiled_circuit: Path,
        path_compiled_circuits: Path,
        figure_of_merit: reward.figure_of_merit = "expected_fidelity",
        logger_level: int = logging.INFO,
    ) -> tuple[tuple[list[Any], Any], str, list[float]]:
        """Handles to create a training sample from a given file.

        Args:
            file (Path): The name of the file to be used for training
            figure_of_merit (reward.reward_functions, optional): The figure of merit to be used for compilation. Defaults to "expected_fidelity".
            path_uncompiled_circuit (Path): The path to the directory containing the uncompiled circuits. Defaults to None.
            path_compiled_circuits (Path): The path to the directory containing the compiled circuits. Defaults to None.
            logger_level (int, optional): The level of the logger. Defaults to logging.INFO.

        Returns:
            tuple[tuple[list[Any], Any], str, list[float]]: Training_sample, circuit_name, scores
        """
        logger.setLevel(logger_level)

        if ".qasm" not in str(file):
            raise RuntimeError("File is not a qasm file: " + str(file))

        LUT = ml.helper.get_index_to_device_LUT()
        logger.debug("Checking " + str(file))
        scores = [-1.0 for _ in range(len(LUT))]
        all_relevant_files = path_compiled_circuits.glob(str(file).split(".")[0] + "*")

        for filename in all_relevant_files:
            filename_str = str(filename)
            if (str(file).split(".")[0] + "_" + figure_of_merit + "_") not in filename_str and filename_str.endswith(
                ".qasm"
            ):
                continue
            comp_path_index = int(filename_str.split("_")[-1].split(".")[0])
            device = LUT[comp_path_index]
            qc = QuantumCircuit.from_qasm_file(filename_str)
            if figure_of_merit == "critical_depth":
                score = reward.crit_depth(qc)
            elif figure_of_merit == "expected_fidelity":
                score = reward.expected_fidelity(qc, device)
            scores[comp_path_index] = score

        num_not_empty_entries = 0
        for i in range(len(LUT)):
            if scores[i] != -1.0:
                num_not_empty_entries += 1

        if num_not_empty_entries == 0:
            logger.warning("no compiled circuits found for:" + str(file))

        feature_vec = ml.helper.create_feature_dict(str(path_uncompiled_circuit / file))
        training_sample = (list(feature_vec.values()), np.argmax(scores))
        circuit_name = str(file).split(".")[0]
        return (training_sample, circuit_name, scores)

    def train_random_forest_classifier(
        self, figure_of_merit: reward.figure_of_merit = "expected_fidelity", visualize_results: bool = False
    ) -> bool:
        """Trains a random forest classifier for the given figure of merit.

        Args:
            figure_of_merit (reward.reward_functions, optional): The figure of merit to be used for training. Defaults to "expected_fidelity".
            visualize_results (bool, optional): Whether to visualize the results. Defaults to False.

        Returns:
            bool: Whether the training was successful.
        """

        training_data = self.get_prepared_training_data(figure_of_merit, save_non_zero_indices=True)

        scores_filtered = [training_data.scores_list[i] for i in training_data.indices_test]
        names_filtered = [training_data.names_list[i] for i in training_data.indices_test]

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
        clf = GridSearchCV(clf, tree_param, cv=5, n_jobs=8).fit(training_data.X_train, training_data.y_train)

        if visualize_results:
            y_pred = np.array(list(clf.predict(training_data.X_test)))
            res, _ = self.calc_performance_measures(scores_filtered, y_pred, training_data.y_test)
            self.plot_eval_histogram(res, filename="RandomForestClassifier")

            logger.info("Best Accuracy: " + str(clf.best_score_))
            top3 = (res.count(1) + res.count(2) + res.count(3)) / len(res)
            logger.info("Top 3: " + str(top3))
            logger.info("Feature Importance: " + str(clf.best_estimator_.feature_importances_))
            self.plot_eval_all_detailed_compact_normed(names_filtered, scores_filtered, y_pred, training_data.y_test)

        self.set_classifier(clf.best_estimator_)
        ml.helper.save_classifier(clf.best_estimator_, figure_of_merit)
        logger.info("Random Forest classifier is trained and saved.")

        return self.clf is not None

    def get_prepared_training_data(
        self, figure_of_merit: reward.figure_of_merit, save_non_zero_indices: bool = False
    ) -> ml.helper.TrainingData:
        """Prepares the training data for the given figure of merit.

        Args:
            figure_of_merit (reward.reward_functions): The figure of merit to be used for training.
            save_non_zero_indices (bool, optional): Whether to save the non zero indices. Defaults to False.

        Returns:
            ml.helper.TrainingData: The prepared training data.
        """
        training_data, names_list, raw_scores_list = ml.helper.load_training_data(figure_of_merit)
        unzipped_training_data_X, unzipped_training_data_Y = zip(*training_data)
        scores_list: list[list[float]] = [[] for _ in range(len(raw_scores_list))]
        X_raw = list(unzipped_training_data_X)
        X_list: list[list[float]] = [[] for _ in range(len(X_raw))]
        y_list = list(unzipped_training_data_Y)
        for i in range(len(X_raw)):
            X_list[i] = list(X_raw[i])
            scores_list[i] = list(raw_scores_list[i])

        X, y, indices = np.array(X_list), np.array(y_list), np.array(range(len(y_list)))

        # Store all non zero feature indices
        non_zero_indices = [i for i in range(len(X[0])) if sum(X[:, i]) > 0]
        X = X[:, non_zero_indices]

        if save_non_zero_indices:
            data = np.asarray(non_zero_indices)
            np.save(
                ml.helper.get_path_trained_model(figure_of_merit, return_non_zero_indices=True),
                data,
            )

        (
            X_train,
            X_test,
            y_train,
            y_test,
            indices_train,
            indices_test,
        ) = train_test_split(X, y, indices, test_size=0.3, random_state=5)

        return ml.helper.TrainingData(
            X_train,
            X_test,
            y_train,
            y_test,
            indices_train,
            indices_test,
            names_list,
            scores_list,
        )

    def calc_performance_measures(
        self,
        scores_filtered: list[list[float]],
        y_pred: np.ndarray[Any, np.dtype[np.float64]],
        y_test: np.ndarray[Any, np.dtype[np.float64]],
    ) -> tuple[list[int], list[float]]:
        """Method to generate the performance measures for a trained classifier

        Args:
            scores_filtered (list[list[float]]): The scores filtered for the respectively predicted indices of all training data
            y_pred (np.ndarray[Any, np.dtype[np.float64]]): The predicted labels
            y_test (np.ndarray[Any, np.dtype[np.float64]]): The actual labels

        Returns:
            tuple[list[int], list[float]]: The ranks and the relative scores
        """

        res = []
        relative_scores = []
        for i in range(len(y_pred)):
            assert np.argmax(scores_filtered[i]) == y_test[i]
            predicted_score = scores_filtered[i][y_pred[i]]
            relative_scores.append(predicted_score - np.max(scores_filtered[i]))
            score = list(np.sort(scores_filtered[i])[::-1]).index(predicted_score)
            res.append(score + 1)

        assert len(res) == len(y_pred)

        return res, relative_scores

    def plot_eval_histogram(self, res: list[int], filename: str = "histogram", color: str = "#21918c") -> None:
        """Method to generate the histogram for the evaluation scores

        Args:
            res (list[int]): The ranks of the predictions
            filename (str, optional): The filename of the histogram. Defaults to "histogram".
            color (str, optional): The color of the histogram. Defaults to "#21918c".

        """

        plt.figure(figsize=(10, 5))

        num_of_comp_paths = len(ml.helper.get_index_to_device_LUT())
        plt.bar(
            list(range(0, num_of_comp_paths, 1)),
            height=[res.count(i) / len(res) for i in range(1, num_of_comp_paths + 1, 1)],
            width=0.90,
            color=color,
        )

        plt.xticks(
            list(range(0, num_of_comp_paths, 1)),
            [i if i % 2 == 1 else "" for i in range(1, num_of_comp_paths + 1, 1)],
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
        plt.savefig(result_path / (filename + ".pdf"), bbox_inches="tight")
        plt.show()

    def plot_eval_all_detailed_compact_normed(
        self,
        names_list: list[Any],
        scores_filtered: list[Any],
        y_pred: np.ndarray[Any, np.dtype[np.float64]],
        y_test: np.ndarray[Any, np.dtype[np.float64]],
        color_all: str = "#21918c",
        color_pred: str = "#440154",
    ) -> None:
        """Method to generate the plot for the evaluation scores of all training data

        Args:
            names_list (list[Any]): The names of all training data
            scores_filtered (list[Any]): The scores filtered for the respectively predicted indices of all training data
            y_pred (np.ndarray[Any, np.dtype[np.float64]]): The predicted labels
            y_test (np.ndarray[Any, np.dtype[np.float64]]): The actual labels
            color_all (str, optional): The color of the evaluation scores of all training data. Defaults to "#21918c".
            color_pred (str, optional): The color of the evaluation scores of the predicted training data. Defaults to "#440154".

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
                plt.plot(i, tmp_res[j] / max_score, "b.", alpha=1.0, markersize=1.7, color=color_all)

            plt.plot(
                i,
                tmp_res[y_pred_sorted_accordingly[i]] / max_score,
                color=color_pred,
                marker=".",
                linestyle="None",
            )

        plt.xticks(
            list(range(0, len(scores_filtered), 30)),
            [qubit_list_sorted[i] for i in range(0, len(scores_filtered), 30)],
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
        plt.savefig(result_path / "y_pred_eval_normed.pdf", bbox_inches="tight")

    def predict_probs(self, qasm_str_or_path: str | QuantumCircuit, figure_of_merit: reward.figure_of_merit) -> int:
        """Returns the probabilities for all supported quantum devices to be the most suitable one for the given quantum circuit.

        Args:
            qasm_str_or_path (str | QuantumCircuit): The qasm string or path to the qasm file
            figure_of_merit (reward.reward_functions): The figure of merit to be used for prediction

        Returns:
            int: The index of the predicted compilation option
        """

        if self.clf is None:
            path = ml.helper.get_path_trained_model(figure_of_merit)
            if path.is_file():
                self.clf = load(str(path))
            else:
                error_msg = "Classifier is neither trained nor saved."
                raise FileNotFoundError(error_msg)

        feature_dict = ml.helper.create_feature_dict(qasm_str_or_path)
        feature_vector = list(feature_dict.values())

        path = ml.helper.get_path_trained_model(figure_of_merit, return_non_zero_indices=True)
        non_zero_indices = np.load(path, allow_pickle=True)
        feature_vector = [feature_vector[i] for i in non_zero_indices]

        return cast(int, self.clf.predict_proba([feature_vector])[0])  # type: ignore[attr-defined]
