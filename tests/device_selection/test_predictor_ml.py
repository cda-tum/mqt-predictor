from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from mqt.bench import benchmark_generator
from mqt.predictor import ml, reward


@patch("matplotlib.pyplot.show")
def test_predict(mock_show: Any) -> None:  # noqa: ARG001
    path = ml.helper.get_path_trained_model() / "trained_clf_fidelity.joblib"
    assert path.is_file()
    filename = "test_qasm.qasm"
    figure_of_merit: reward.reward_functions = "fidelity"
    qc = benchmark_generator.get_benchmark("dj", 1, 8)
    qc.qasm(filename=filename)
    predictor = ml.Predictor()
    prediction = predictor.predict(filename, figure_of_merit=figure_of_merit)
    assert 0 <= prediction < len(ml.helper.get_index_to_device_LUT())
    prediction = predictor.predict(qc.qasm(), figure_of_merit=figure_of_merit)
    assert 0 <= prediction < len(ml.helper.get_index_to_device_LUT())
    with pytest.raises(ValueError, match="Invalid input for 'qc' parameter"):
        predictor.predict("Error Test", figure_of_merit=figure_of_merit)

    predictor.clf = None
    prediction = predictor.predict(filename, figure_of_merit=figure_of_merit)
    Path(filename).unlink()
    assert 0 <= prediction < len(ml.helper.get_index_to_device_LUT())


@patch("matplotlib.pyplot.show")
def test_train_random_forest_classifier(mock_pyplot: Any) -> None:  # noqa: ARG001
    predictor = ml.Predictor()
    assert predictor.clf is None
    predictor.train_random_forest_classifier(visualize_results=False)
    if Path("non_zero_indices.npy").exists():
        Path("non_zero_indices.npy").unlink()

    assert predictor.clf is not None


def test_generate_compiled_circuits() -> None:
    predictor = ml.Predictor()
    source_path = Path()
    target_path = Path("test_compiled_circuits")
    if not target_path.exists():
        target_path.mkdir()
    figure_of_merit: reward.reward_functions = "fidelity"

    qc = benchmark_generator.get_benchmark("dj", 1, 3)
    qasm_path = Path("test.qasm")
    qc.qasm(filename=str(qasm_path))
    predictor.generate_compiled_circuits(source_path, target_path)
    assert any(file.suffix == ".qasm" for file in target_path.iterdir())

    res = predictor.generate_training_sample(
        file=str(qasm_path),
        figure_of_merit=figure_of_merit,
        path_uncompiled_circuit=str(source_path),
        path_compiled_circuits=str(target_path),
    )
    assert not isinstance(res, bool)
    training_sample, circuit_name, scores = res
    assert training_sample
    assert circuit_name
    assert scores

    (
        training_data,
        name_list,
        scores_list,
    ) = predictor.generate_trainingdata_from_qasm_files(figure_of_merit, str(source_path), str(target_path))
    assert training_data
    assert name_list
    assert scores_list

    if target_path.exists():
        for file in target_path.iterdir():
            file.unlink()
        target_path.rmdir()

    if qasm_path.exists():
        qasm_path.unlink()
