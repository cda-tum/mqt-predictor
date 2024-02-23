from __future__ import annotations

from pathlib import Path

import numpy as np

from mqt.bench import benchmark_generator
from mqt.bench.devices import get_available_devices
from mqt.predictor import ml, reward


def test_predict() -> None:
    path = ml.helper.get_path_trained_model(figure_of_merit="expected_fidelity")
    assert path.is_file()
    filename = "test_qasm.qasm"
    figure_of_merit: reward.figure_of_merit = "expected_fidelity"
    qc = benchmark_generator.get_benchmark("dj", 1, 8)
    qc.qasm(filename=filename)
    predictor = ml.Predictor()
    predictions = predictor.predict_probs(filename, figure_of_merit=figure_of_merit)
    assert predictor.clf is not None
    classes = predictor.clf.classes_  # type: ignore[unreachable]
    predicted_device_indices = classes[np.argsort(predictions)[::-1]]
    devices = get_available_devices()
    assert all(0 <= i < len(devices) for i in predicted_device_indices)
    predictions = predictor.predict_probs(qc.qasm(), figure_of_merit=figure_of_merit)
    predicted_device_indices = classes[np.argsort(predictions)[::-1]]
    assert all(0 <= i < len(devices) for i in predicted_device_indices)
    Path(filename).unlink()


def test_train_random_forest_classifier() -> None:
    predictor = ml.Predictor()
    assert predictor.clf is None
    predictor.train_random_forest_classifier(visualize_results=False)

    assert predictor.clf is not None


def test_compile_all_circuits_for_dev_and_fom() -> None:
    predictor = ml.Predictor()
    source_path = Path()
    target_path = Path("test_compiled_circuits")
    if not target_path.exists():
        target_path.mkdir()
    figure_of_merit: reward.figure_of_merit = "expected_fidelity"

    qc = benchmark_generator.get_benchmark("dj", 1, 3)
    qasm_path = Path("test.qasm")
    qc.qasm(filename=str(qasm_path))
    predictor.compile_all_circuits_devicewise(
        device_name="ionq_harmony",
        timeout=100,
        figure_of_merit=figure_of_merit,
        source_path=source_path,
        target_path=target_path,
    )
    assert any(file.suffix == ".qasm" for file in target_path.iterdir())

    training_sample, circuit_name, scores = predictor.generate_training_sample(
        file=qasm_path,
        figure_of_merit=figure_of_merit,
        path_uncompiled_circuit=source_path,
        path_compiled_circuits=target_path,
    )
    assert training_sample
    assert circuit_name is not None
    assert any(score != -1 for score in scores)

    (
        training_data,
        name_list,
        scores_list,
    ) = predictor.generate_trainingdata_from_qasm_files(figure_of_merit, source_path, target_path)
    assert len(training_data) > 0
    assert len(name_list) > 0
    assert len(scores_list) > 0

    if target_path.exists():
        for file in target_path.iterdir():
            file.unlink()
        target_path.rmdir()

    if qasm_path.exists():
        qasm_path.unlink()
