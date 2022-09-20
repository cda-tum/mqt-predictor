import distutils.sysconfig

from predictor.driver import Predictor
import os
from mqt.bench import benchmark_generator
import pytest
from unittest.mock import patch
from predictor.src import utils


@patch("matplotlib.pyplot.show")
def test_train_decision_tree_classifier(mock_show):
    (
        training_data,
        name_list,
        scores_list,
    ) = utils.load_trainig_data()
    X, y = zip(*training_data)
    res = Predictor.train_decision_tree_classifier(X, y, name_list, scores_list)
    assert not res is None


@patch("matplotlib.pyplot.show")
def test_predict(mock_show):
    assert os.path.isfile("decision_tree_classifier.joblib")
    filename = "test_qasm.qasm"
    benchmark_generator.get_one_benchmark("dj", 1, 8).qasm(filename=filename)
    prediction = Predictor.predict(filename)
    assert prediction >= 0 and prediction < len(utils.get_index_to_comppath_LUT())
    prediction = Predictor.predict("fail test")
    assert not prediction

    Predictor._clf = None
    prediction = Predictor.predict(filename)
    os.remove(filename)
    assert prediction >= 0 and prediction < len(utils.get_index_to_comppath_LUT())


@pytest.mark.parametrize(
    "comp_path", [i for i in range(len(utils.get_index_to_comppath_LUT()))]
)
def test_compilation_paths(comp_path):
    qc_qasm = benchmark_generator.get_one_benchmark("dj", 1, 2).qasm()
    res = Predictor.compile_predicted_compilation_path(qc_qasm, comp_path)
    assert res
    qc = benchmark_generator.get_one_benchmark("dj", 1, 2)
    tmp_filename = "test.qasm"
    qc.qasm(filename=tmp_filename)
    res = Predictor.compile_predicted_compilation_path(tmp_filename, comp_path)
    assert res
    if os.path.isfile(tmp_filename):
        os.remove(tmp_filename)


def test_compile_all_circuits_for_qc():
    qc = benchmark_generator.get_one_benchmark("dj", 1, 2)
    tmp_filename = "test.qasm"
    qc.qasm(filename=tmp_filename)
    assert Predictor.compile_all_circuits_for_qc(
        filename=tmp_filename, source_path=".", target_directory="./comp_test/"
    )
    if os.path.isfile(tmp_filename):
        os.remove(tmp_filename)
