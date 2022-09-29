import os
from unittest.mock import patch

import pytest

from mqt.bench import benchmark_generator
from mqt.predictor.driver import Predictor
from mqt.predictor.src import utils


@patch("matplotlib.pyplot.show")
def test_predict(mock_show):
    assert os.path.isfile("decision_tree_classifier.joblib")
    filename = "test_qasm.qasm"
    benchmark_generator.get_one_benchmark("dj", 1, 8).qasm(filename=filename)
    predictor = Predictor()
    prediction = predictor.predict(filename)
    assert prediction >= 0 and prediction < len(utils.get_index_to_comppath_LUT())
    prediction = predictor.predict("fail test")
    assert not prediction

    predictor.clf = None
    prediction = predictor.predict(filename)
    os.remove(filename)
    assert prediction >= 0 and prediction < len(utils.get_index_to_comppath_LUT())


@pytest.mark.parametrize(
    "comp_path", list(range(len(utils.get_index_to_comppath_LUT())))
)
def test_compilation_paths(comp_path):
    predictor = Predictor()
    qc_qasm = benchmark_generator.get_one_benchmark("dj", 1, 2).qasm()
    res = predictor.compile_predicted_compilation_path(qc_qasm, comp_path)
    assert res
    qc = benchmark_generator.get_one_benchmark("dj", 1, 2)
    tmp_filename = "test.qasm"
    qc.qasm(filename=tmp_filename)
    res = predictor.compile_predicted_compilation_path(tmp_filename, comp_path)
    assert res
    if os.path.isfile(tmp_filename):
        os.remove(tmp_filename)


def test_compile_all_circuits_for_qc():
    qc = benchmark_generator.get_one_benchmark("dj", 1, 2)
    tmp_filename = "test.qasm"
    qc.qasm(filename=tmp_filename)
    predictor = Predictor()
    assert predictor.compile_all_circuits_for_qc(
        filename=tmp_filename,
        source_path="",
        target_directory="./training_samples_compiled/",
    )
    if os.path.isfile(tmp_filename):
        os.remove(tmp_filename)
