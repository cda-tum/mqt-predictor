from __future__ import annotations

import pytest

from mqt.bench import benchmark_generator
from mqt.predictor import ml


def test_load_training_data() -> None:
    assert ml.helper.load_training_data() is not None

    with pytest.raises(FileNotFoundError, match="Training data not found. Please run the training script first."):
        ml.helper.load_training_data("false_input")  # type: ignore[arg-type]


def test_save_training_data() -> None:
    training_data, names_list, scores_list = ml.helper.load_training_data()
    ml.helper.save_training_data(training_data, names_list, scores_list, "expected_fidelity")


def test_create_feature_dict() -> None:
    qc = benchmark_generator.get_benchmark("dj", 1, 3)
    feature_vector = ml.helper.create_feature_dict(qc)
    assert feature_vector is not None

    with pytest.raises(ValueError, match="Invalid input for 'qc' parameter."):
        ml.helper.create_feature_dict("false_input")


def test_get_openqasm_gates() -> None:
    assert ml.helper.get_openqasm_gates() is not None


def test_get_path_training_circuits() -> None:
    path = ml.helper.get_path_training_circuits()
    assert path.exists()


def test_get_path_training_circuits_compiled() -> None:
    path = ml.helper.get_path_training_circuits_compiled()
    assert path.exists()


def test_get_path_training_data() -> None:
    path = ml.helper.get_path_training_data()
    assert path.exists()


def test_get_path_trained_model() -> None:
    for figure_of_merit in ["expected_fidelity", "critical_depth"]:
        path = ml.helper.get_path_trained_model(figure_of_merit=figure_of_merit)
        assert path.exists()


def test_get_path_results() -> None:
    for get_ghz_path_results in (True, False):
        path = ml.helper.get_path_results(ghz_results=get_ghz_path_results)
        assert path.exists()
