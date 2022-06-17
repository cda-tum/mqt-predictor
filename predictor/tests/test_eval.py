from predictor.driver import Predictor
import os
from mqt.bench import benchmark_generator
import pytest
from unittest.mock import patch


@patch("matplotlib.pyplot.show")
def test_train_decision_tree_classifier(mock_show):
    (
        training_data,
        name_list,
        scores_list,
    ) = Predictor.generate_trainingdata_from_qasm_files(
        folder_path="./test_source/", compiled_path="./test_compiled_target/"
    )
    X, y = zip(*training_data)
    res = Predictor.train_decision_tree_classifier(X, y, name_list, scores_list)
    assert not res is None


@patch("matplotlib.pyplot.show")
def test_predict(mock_show):
    assert os.path.isfile("decision_tree_classifier.joblib")
    filename = "test_qasm.qasm"
    benchmark_generator.get_one_benchmark("dj", 1, 8).qasm(filename=filename)
    prediction = Predictor.predict(filename)
    assert prediction >= 0 and prediction < 19
    prediction = Predictor.predict("fail test")
    assert not prediction

    Predictor._clf = None
    prediction = Predictor.predict(filename)
    os.remove(filename)
    assert prediction >= 0 and prediction < 19


@pytest.mark.parametrize("comp_path", [i for i in range(19)])
def test_compilation_paths(comp_path):
    qc_qasm = benchmark_generator.get_one_benchmark("dj", 1, 2).qasm()
    qc = benchmark_generator.get_one_benchmark("dj", 1, 2)
    res = Predictor.compile_predicted_compilation_path(qc_qasm, comp_path)
    assert not res is None
    res = Predictor.compile_predicted_compilation_path(qc_qasm, 20)
    assert res is None
