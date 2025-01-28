"""Tests for the machine learning device selection predictor module."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
from qiskit.qasm2 import dump

import mqt.bench.devices
from mqt.bench import get_benchmark
from mqt.predictor import ml, rl


def test_train_and_predictor_random_forest_classifier() -> None:
    """Test the training of a random forest classifier.

    This test must be executed prior to any prediction to make sure the model is trained using the latest scikit-learn version.
    """

    # create the ML Predictor
    predictor = ml.Predictor(devices=["ionq_harmony"], figure_of_merit="expected_fidelity")
    source_path = Path()
    target_path = Path("test_compiled_circuits")
    if not target_path.exists():
        target_path.mkdir()

    # create uncompiled training data
    qc = get_benchmark("dj", 1, 3)
    qasm_path = Path("test.qasm")
    with Path(qasm_path).open("w", encoding="utf-8") as f:
        dump(qc, f)

    # generate compiled circuits using trained RL model
    if sys.platform == "win32":
        with pytest.warns(RuntimeWarning, match=re.escape("Timeout is not supported on Windows.")):
            predictor.generate_compiled_circuits(timeout=600, target_path=target_path, source_path=source_path)
    else:
        predictor.generate_compiled_circuits(timeout=600)

    training_data, names_list, scores_list = predictor.generate_trainingdata_from_qasm_files()
    assert len(training_data) > 0
    assert len(names_list) > 0
    assert len(scores_list) > 0

    # train the random forest classifier based on created training data
    predictor.train_random_forest_classifier(save_classifier=True)

    # save the training data and delete it afterwards
    ml.helper.save_training_data(training_data, names_list, scores_list, "test")
    for file in ["training_data_test.npy", "names_list_test.npy", "scores_list_test.npy"]:
        path = ml.helper.get_path_training_data() / "training_data_aggregated" / file
        assert path.exists()
        path.unlink()

    # delete used RL model to save storage
    model_path = Path(rl.helper.get_path_trained_model() / ("model_expected_fidelity_ionq_harmony.zip"))
    model_path.unlink()

    # delete the uncompiled and compiled circuits
    delete_path(target_path)
    if qasm_path.exists():
        qasm_path.unlink()

    # test the prediction for given Qiskit.QuantumCircuit and QASM file
    qc = get_benchmark("ghz", 1, 3)
    predicted_dev = ml.predict_device_for_figure_of_merit(qc)
    assert predicted_dev in mqt.bench.devices.get_available_devices()

    file = Path("test_qasm.qasm")
    qc = get_benchmark("dj", 1, 3)
    with file.open("w", encoding="utf-8") as f:
        dump(qc, f)

    predicted_dev = ml.predict_device_for_figure_of_merit(
        file,
    )
    assert predicted_dev in mqt.bench.devices.get_available_devices()

    with pytest.raises(
        FileNotFoundError, match=re.escape("The ML model is not trained yet. Please train the model before using it.")
    ):
        ml.predict_device_for_figure_of_merit(qc, figure_of_merit="false_input")  # type: ignore[arg-type]


def test_performance_measures() -> None:
    """Test the calculation of the performance measures for a given set of scores and labels."""
    predictor = ml.Predictor(figure_of_merit="expected_fidelity")

    training_data = predictor.get_prepared_training_data()

    y_test = training_data.y_test
    indices_test = training_data.indices_test
    names_list = training_data.names_list
    scores_list = training_data.scores_list

    assert len(y_test) > 0
    assert len(indices_test) > 0
    assert len(names_list) > 0
    assert len(scores_list) > 0

    scores_filtered = [scores_list[i] for i in indices_test]
    names_filtered = [names_list[i] for i in indices_test]

    # Test calc_performance_measures
    res, relative_scores = predictor.calc_performance_measures(scores_filtered, y_test, y_test)
    assert all(res)
    assert not any(relative_scores)

    # Test generate_eval_histogram
    predictor.generate_eval_histogram(res, show_plot=False)
    histogram_path = Path("results/histogram.pdf")
    assert histogram_path.is_file(), "File does not exist"
    histogram_path.unlink()

    # Test generate_eval_all_datapoints
    predictor.generate_eval_all_datapoints(names_filtered, scores_filtered, y_test, y_test)
    result_path = Path("results/y_pred_eval_normed.pdf")
    assert result_path.is_file(), "File does not exist"
    result_path.unlink()


def delete_path(path: Path) -> None:
    """Delete a path recursively."""
    if path.exists():
        for item in path.iterdir():
            if item.is_dir():
                delete_path(item)  # Recursive deletion
            else:
                item.unlink()  # Delete file
        path.rmdir()  # Remove empty directory
