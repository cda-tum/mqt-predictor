"""This module contains the Predictor class, which is used to predict the most suitable quantum device for a given quantum circuit."""

from __future__ import annotations

import logging
import sys
import zipfile
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

if sys.version_info >= (3, 11) and TYPE_CHECKING:  # pragma: no cover
    from typing import assert_never
else:
    from typing_extensions import assert_never

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed, load
from qiskit import QuantumCircuit
from qiskit.qasm2 import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

from mqt.bench.devices import get_available_devices, get_device_by_name
from mqt.predictor import ml, reward, rl, utils

if TYPE_CHECKING:
    from numpy._typing import NDArray

plt.rcParams["font.family"] = "Times New Roman"

logger = logging.getLogger("mqt-predictor")


class Predictor:
    """The Predictor class is used to predict the most suitable quantum device for a given quantum circuit."""

    def __init__(
        self,
        figure_of_merit: reward.figure_of_merit = "expected_fidelity",
        devices: list[str] | None = None,
        logger_level: int = logging.INFO,
    ) -> None:
        """Initializes the Predictor class.

        Arguments:
            figure_of_merit: The figure of merit to be used for training.
            devices: The devices to be used for training. Defaults to None. If None, all available devices from MQT Bench are used.
            logger_level: The level of the logger. Defaults to logging.INFO.

        """
        logger.setLevel(logger_level)

        self.clf = None
        self.figure_of_merit = figure_of_merit
        if devices is None:
            self.devices = get_available_devices()
        else:
            self.devices = [get_device_by_name(device) for device in devices]
        self.devices.sort(
            key=lambda x: x.name
        )  # sorting is necessary to determine the ground truth label later on when generating the training data

    def set_classifier(self, clf: RandomForestClassifier) -> None:
        """Sets the classifier to the given classifier."""
        self.clf = clf

    def compile_all_circuits_devicewise(
        self,
        device_name: str,
        timeout: int,
        source_path: Path | None = None,
        target_path: Path | None = None,
        logger_level: int = logging.INFO,
    ) -> None:
        """Compiles all circuits in the given directory with the given timeout and saves them in the given directory.

        Arguments:
            device_name: The name of the device to be used for compilation.
            timeout: The timeout in seconds for the compilation of a single circuit.
            source_path: The path to the directory containing the circuits to be compiled. Defaults to None.
            target_path: The path to the directory where the compiled circuits should be saved. Defaults to None.
            logger_level: The level of the logger. Defaults to logging.INFO.
        """
        logger.setLevel(logger_level)

        logger.info("Processing: " + device_name + " for " + self.figure_of_merit)
        rl_pred = rl.Predictor(figure_of_merit=self.figure_of_merit, device_name=device_name)

        dev = get_device_by_name(device_name)
        dev_max_qubits = dev.num_qubits

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
                str(filename).split("/")[-1].split(".qasm")[0] + "_" + self.figure_of_merit + "-" + dev.name
            )
            if (Path(target_path) / (target_filename + ".qasm")).exists():
                continue
            try:
                res = utils.timeout_watcher(rl.qcompile, [qc, self.figure_of_merit, device_name, rl_pred], timeout)
                if isinstance(res, tuple):
                    compiled_qc = res[0]
                    with Path(target_path / (target_filename + ".qasm")).open("w", encoding="utf-8") as f:
                        dump(compiled_qc, f)

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

        Arguments:
            source_path: The path to the directory containing the circuits to be compiled. Defaults to None.
            target_path: The path to the directory where the compiled circuits should be saved. Defaults to None.
            timeout: The timeout in seconds for the compilation of a single circuit. Defaults to 600.
        """
        if source_path is None:
            source_path = ml.helper.get_path_training_circuits()

        if target_path is None:
            target_path = ml.helper.get_path_training_circuits_compiled()

        path_zip = source_path / "training_data_device_selection.zip"
        if not any(file.suffix == ".qasm" for file in source_path.iterdir()) and path_zip.exists():
            with zipfile.ZipFile(str(path_zip), "r") as zip_ref:
                zip_ref.extractall(source_path)

        target_path.mkdir(exist_ok=True)

        Parallel(n_jobs=-1, verbose=100)(
            delayed(self.compile_all_circuits_devicewise)(device.name, timeout, source_path, target_path, logger.level)
            for device in self.devices
        )

    def generate_trainingdata_from_qasm_files(
        self,
        path_uncompiled_circuits: Path | None = None,
        path_compiled_circuits: Path | None = None,
    ) -> tuple[list[NDArray[np.float64]], list[str], list[NDArray[np.float64]]]:
        """Handles to create training data from all generated training samples.

        Arguments:
            figure_of_merit: The figure of merit to be used for training.
            path_uncompiled_circuits: The path to the directory containing the uncompiled circuits. Defaults to None.
            path_compiled_circuits: The path to the directory containing the compiled circuits. Defaults to None.

        Returns:
            The training data, consisting of training_data, name_list, scores_list

        """
        if not path_uncompiled_circuits:
            path_uncompiled_circuits = ml.helper.get_path_training_circuits()

        if not path_compiled_circuits:
            path_compiled_circuits = ml.helper.get_path_training_circuits_compiled()

        # init resulting list (feature vector, name, scores)
        training_data = []
        name_list = []
        scores_list = []

        results = Parallel(n_jobs=1, verbose=100)(
            delayed(self.generate_training_sample)(
                filename.name,
                path_uncompiled_circuits,
                path_compiled_circuits,
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
        logger_level: int = logging.INFO,
    ) -> tuple[tuple[list[Any], Any], str, list[float]]:
        """Handles to create a training sample from a given file.

        Arguments:
            file: The name of the file to be used for training.
            path_uncompiled_circuit: The path to the directory containing the uncompiled circuits. Defaults to None.
            path_compiled_circuits: The path to the directory containing the compiled circuits. Defaults to None.
            logger_level: The level of the logger. Defaults to logging.INFO.

        Returns:
            Training_sample, circuit_name, scores
        """
        logger.setLevel(logger_level)

        if ".qasm" not in str(file):
            raise RuntimeError("File is not a qasm file: " + str(file))

        logger.debug("Checking " + str(file))
        scores = {dev.name: -1.0 for dev in self.devices}
        all_relevant_files = path_compiled_circuits.glob(str(file).split(".")[0] + "*")

        for filename in all_relevant_files:
            filename_str = str(filename)
            if (str(file).split(".")[0] + "_" + self.figure_of_merit) not in filename_str and filename_str.endswith(
                ".qasm"
            ):
                continue
            dev_name = filename_str.split("-")[-1].split(".")[0]
            if dev_name not in [dev.name for dev in self.devices]:
                continue
            device = get_device_by_name(dev_name)
            qc = QuantumCircuit.from_qasm_file(filename_str)
            if self.figure_of_merit == "critical_depth":
                score = reward.crit_depth(qc)
            elif self.figure_of_merit == "expected_fidelity":
                score = reward.expected_fidelity(qc, device)
            elif self.figure_of_merit == "estimated_success_probability":
                score = reward.estimated_success_probability(qc, device)
            else:
                assert_never(self.figure_of_merit)
            scores[dev_name] = score

        num_not_empty_entries = 0
        for dev in self.devices:
            if scores[dev.name] != -1.0:
                num_not_empty_entries += 1

        if num_not_empty_entries == 0:
            logger.warning("no compiled circuits found for:" + str(file))

        feature_vec = ml.helper.create_feature_dict(path_uncompiled_circuit / file)
        scores_list = list(scores.values())
        target_label = np.argmax(scores_list)

        training_sample = (list(feature_vec.values()), target_label)
        circuit_name = str(file).split(".")[0]
        return training_sample, circuit_name, scores_list

    def train_random_forest_classifier(
        self,
        visualize_results: bool = False,
        save_classifier: bool = True,
    ) -> bool:
        """Trains a random forest classifier for the given figure of merit.

        Arguments:
            visualize_results: Whether to visualize the results. Defaults to False.
            save_classifier: Whether to save the classifier. Defaults to True.

        Returns:
            True when the training was successful, False otherwise.
        """
        training_data = self.get_prepared_training_data(save_non_zero_indices=True)

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
        clf = GridSearchCV(clf, tree_param, cv=2, n_jobs=8).fit(training_data.X_train, training_data.y_train)

        if visualize_results:
            y_pred = np.array(list(clf.predict(training_data.X_test)))
            res, _ = self.calc_performance_measures(scores_filtered, y_pred, training_data.y_test)
            self.generate_eval_histogram(res, filename="RandomForestClassifier")

            logger.info("Best Accuracy: " + str(clf.best_score_))
            top3 = (res.count(1) + res.count(2) + res.count(3)) / len(res)
            logger.info("Top 3: " + str(top3))
            logger.info("Feature Importance: " + str(clf.best_estimator_.feature_importances_))
            self.generate_eval_all_datapoints(names_filtered, scores_filtered, y_pred, training_data.y_test)

        self.set_classifier(clf.best_estimator_)
        if save_classifier:
            dump(clf, str(ml.helper.get_path_trained_model(self.figure_of_merit)))
        logger.info("Random Forest classifier is trained and saved.")

        return self.clf is not None

    def get_prepared_training_data(self, save_non_zero_indices: bool = False) -> ml.helper.TrainingData:
        """Prepares the training data for the given figure of merit.

        Arguments:
            save_non_zero_indices: Whether to save the non zero indices. Defaults to False.

        Returns:
            The prepared training data.
        """
        training_data, names_list, raw_scores_list = self.load_training_data()
        unzipped_training_data_x, unzipped_training_data_y = zip(*training_data, strict=False)
        scores_list: list[list[float]] = [[] for _ in range(len(raw_scores_list))]
        x_raw = list(unzipped_training_data_x)
        x_list: list[list[float]] = [[] for _ in range(len(x_raw))]
        y_list = list(unzipped_training_data_y)
        for i in range(len(x_raw)):
            x_list[i] = list(x_raw[i])
            scores_list[i] = list(raw_scores_list[i])

        x, y, indices = (
            np.array(x_list, dtype=np.float64),
            np.array(y_list, dtype=np.int64),
            np.array(range(len(y_list)), dtype=np.int64),
        )

        # Store all non zero feature indices
        non_zero_indices = [i for i in range(len(x[0])) if sum(x[:, i]) > 0]
        x = x[:, non_zero_indices]

        if save_non_zero_indices:
            data = np.asarray(non_zero_indices, dtype=np.uint64)
            np.save(
                ml.helper.get_path_trained_model(self.figure_of_merit, return_non_zero_indices=True),
                data,
            )

        (
            x_train,
            x_test,
            y_train,
            y_test,
            indices_train,
            indices_test,
        ) = train_test_split(x, y, indices, test_size=0.3, random_state=5)

        return ml.helper.TrainingData(
            x_train,
            x_test,
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
        """Method to generate the performance measures for a trained classifier.

        Arguments:
            scores_filtered: The scores filtered for the respectively predicted indices of all training data.
            y_pred: The predicted labels.
            y_test: The actual labels.

        Returns:
            The ranks and the relative scores.
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

    def generate_eval_histogram(
        self, res: list[int], filename: str = "histogram", color: str = "#21918c", show_plot: bool = True
    ) -> None:
        """Method to generate the histogram for the evaluation scores.

        Arguments:
            res: The ranks of the predictions.
            filename: The filename of the histogram. Defaults to "histogram".
            color: The color of the histogram. Defaults to "#21918c".
            show_plot: Whether to show the plot. Defaults to True. False for testing purposes.

        """
        plt.figure(figsize=(10, 5))

        num_of_comp_paths = len(self.devices)
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
        if show_plot:
            plt.show()

    def generate_eval_all_datapoints(
        self,
        names_list: list[Any],
        scores_filtered: list[Any],
        y_pred: np.ndarray[Any, np.dtype[np.float64]],
        y_test: np.ndarray[Any, np.dtype[np.float64]],
        color_all: str = "#21918c",
        color_pred: str = "#440154",
    ) -> None:
        """Method to generate the plot for the evaluation scores of all training data.

        Arguments:
            names_list: The names of all training data.
            scores_filtered: The scores filtered for the respectively predicted indices of all training data.
            y_pred: The predicted labels.
            y_test: The actual labels.
            color_all: The color of the evaluation scores of all training data. Defaults to "#21918c".
            color_pred: The color of the evaluation scores of the predicted training data. Defaults to "#440154".

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
        ) = zip(*sorted(zip(names_list_num_qubits, scores_filtered, y_pred, strict=False)), strict=False)
        plt.figure(figsize=(17, 8))
        for i in range(len(names_list_num_qubits)):
            tmp_res = scores_filtered_sorted_accordingly[i]
            max_score = max(tmp_res)
            if max_score == 0:
                continue
            for j in range(len(tmp_res)):
                plt.plot(i, tmp_res[j] / max_score, alpha=1.0, markersize=1.7, color=color_all)

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

    def predict_probs(self, qc: Path | QuantumCircuit) -> NDArray[np.float64]:
        """Returns the probabilities for all supported quantum devices to be the most suitable one for the given quantum circuit.

        Arguments:
            qc: The QuantumCircuit or Path to the respective qasm file.

        Returns:
            The probabilities for all supported quantum devices to be the most suitable one for the given quantum circuit.
        """
        if self.clf is None:
            path = ml.helper.get_path_trained_model(self.figure_of_merit)
            if path.is_file():
                self.clf = load(path)

            if self.clf is None:
                error_msg = "The ML model is not trained yet. Please train the model before using it."
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

        feature_dict = ml.helper.create_feature_dict(qc)  # type: ignore[unreachable]
        feature_vector = list(feature_dict.values())

        path = ml.helper.get_path_trained_model(self.figure_of_merit, return_non_zero_indices=True)
        non_zero_indices = np.load(path, allow_pickle=True)
        feature_vector = [feature_vector[i] for i in non_zero_indices]

        return self.clf.predict_proba([feature_vector])[0]

    def save_training_data(
        self,
        training_data: list[NDArray[np.float64]],
        names_list: list[str],
        scores_list: list[NDArray[np.float64]],
    ) -> None:
        """Saves the given training data to the training data folder.

        Arguments:
            training_data: The training data, the names list and the scores list to be saved.
            names_list: The names list of the training data.
            scores_list: The scores list of the training data.
            figure_of_merit: The figure of merit to be used for compilation.
        """
        with resources.as_file(ml.helper.get_path_training_data() / "training_data_aggregated") as path:
            data = np.asarray(training_data, dtype=object)
            np.save(str(path / ("training_data_" + self.figure_of_merit + ".npy")), data)
            data = np.asarray(names_list, dtype=str)
            np.save(str(path / ("names_list_" + self.figure_of_merit + ".npy")), data)
            data = np.asarray(scores_list, dtype=object)
            np.save(str(path / ("scores_list_" + self.figure_of_merit + ".npy")), data)

    def load_training_data(self) -> tuple[list[NDArray[np.float64]], list[str], list[NDArray[np.float64]]]:
        """Loads and returns the training data from the training data folder.

        Arguments:
            figure_of_merit: The figure of merit to be used for compilation. Defaults to "expected_fidelity".

        Returns:
           The training data, the names list and the scores list.
        """
        with resources.as_file(ml.helper.get_path_training_data() / "training_data_aggregated") as path:
            if (
                path.joinpath("training_data_" + self.figure_of_merit + ".npy").is_file()
                and path.joinpath("names_list_" + self.figure_of_merit + ".npy").is_file()
                and path.joinpath("scores_list_" + self.figure_of_merit + ".npy").is_file()
            ):
                training_data = np.load(path / ("training_data_" + self.figure_of_merit + ".npy"), allow_pickle=True)
                names_list = list(np.load(path / ("names_list_" + self.figure_of_merit + ".npy"), allow_pickle=True))
                scores_list = list(np.load(path / ("scores_list_" + self.figure_of_merit + ".npy"), allow_pickle=True))
            else:
                error_msg = "Training data not found. Please run the training script first."
                raise FileNotFoundError(error_msg)

            return training_data, names_list, scores_list
