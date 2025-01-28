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


def test_predictor_with_all_devices() -> None:
    """Test the prediction of the device for a given figure of merit with all available devices."""
    predictor = ml.Predictor(figure_of_merit="expected_fidelity", devices=None)
    assert len(predictor.devices) > 0


def test_entire_setup() -> None:
    """Test the training of a random forest classifier."""
    # create the ML Predictor
    predictor = ml.Predictor(devices=["ionq_harmony"], figure_of_merit="expected_fidelity")

    msg = "Training data not found. Please run the training script first as described in the documentation that can be found at https://mqt.readthedocs.io/projects/predictor/en/latest/Usage.html."
    with pytest.raises(FileNotFoundError, match=re.escape(msg)):
        predictor.load_training_data()

    source_path = Path()
    target_path = Path("test_compiled_circuits")
    if not target_path.exists():
        target_path.mkdir()

    # create uncompiled training data
    qc = get_benchmark("dj", 1, 3)
    qasm_path1 = Path("test1.qasm")
    with Path(qasm_path1).open("w", encoding="utf-8") as f:
        dump(qc, f)
    qc = get_benchmark("ghz", 1, 3)
    qasm_path2 = Path("test2.qasm")
    with Path(qasm_path2).open("w", encoding="utf-8") as f:
        dump(qc, f)

    # generate compiled circuits using trained RL model
    if sys.platform == "win32":
        with pytest.warns(RuntimeWarning, match=re.escape("Timeout is not supported on Windows.")):
            predictor.generate_compiled_circuits(
                timeout=600, target_path=target_path, source_path=source_path, num_workers=1
            )
    else:
        predictor.generate_compiled_circuits(
            timeout=600, target_path=target_path, source_path=source_path, num_workers=1
        )

    training_data, names_list, scores_list = predictor.generate_trainingdata_from_qasm_files(
        path_uncompiled_circuits=source_path, path_compiled_circuits=target_path
    )
    assert len(training_data) > 0
    assert len(names_list) > 0
    assert len(scores_list) > 0

    # save the training data and delete it afterwards
    predictor.save_training_data(training_data, names_list, scores_list)
    for file in [
        "training_data_expected_fidelity.npy",
        "names_list_expected_fidelity.npy",
        "scores_list_expected_fidelity.npy",
    ]:
        path = ml.helper.get_path_training_data() / "training_data_aggregated" / file
        assert path.exists()

    # train the random forest classifier based on created training data
    predictor.train_random_forest_classifier(save_classifier=True)

    # delete used RL model to save storage
    # model_path = Path(rl.helper.get_path_trained_model() / ("model_expected_fidelity_ionq_harmony.zip"))
    # model_path.unlink()

    # delete the uncompiled and compiled circuits
    delete_path(target_path)
    if qasm_path1.exists():
        qasm_path1.unlink()
    if qasm_path2.exists():
        qasm_path2.unlink()

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
        ml.predict_device_for_figure_of_merit(qc=qc, figure_of_merit="false_input")  # type: ignore[arg-type]


def delete_path(path: Path) -> None:
    """Delete a path recursively."""
    if path.exists():
        for item in path.iterdir():
            if item.is_dir():
                delete_path(item)  # Recursive deletion
            else:
                item.unlink()  # Delete file
        path.rmdir()  # Remove empty directory


def test_predict_device_for_figure_of_merit_no_suitable_device() -> None:
    """Test the prediction of the device for a given figure of merit with a wrong device name."""
    num_qubits = 100
    qc = get_benchmark("ghz", 1, num_qubits)
    with pytest.raises(
        ValueError, match=re.escape(f"No suitable device found for the given quantum circuit with {num_qubits} qubits.")
    ):
        ml.predict_device_for_figure_of_merit(qc)
