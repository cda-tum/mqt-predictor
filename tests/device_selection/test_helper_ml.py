from __future__ import annotations

import pytest

from mqt.bench import benchmark_generator
from mqt.predictor import ml, qcompile


def test_get_index_to_device_LUT() -> None:
    expected = {
        0: "ibm_washington",
        1: "ibm_montreal",
        2: "oqc_lucy",
        3: "rigetti_aspen_m2",
        4: "ionq_harmony",
        5: "ionq_aria1",
        6: "quantinuum_h2",
    }
    assert ml.helper.get_index_to_device_LUT() == expected


def test_load_training_data() -> None:
    assert ml.helper.load_training_data() is not None

    with pytest.raises(FileNotFoundError):
        ml.helper.load_training_data("false_input")


def test_save_training_data() -> None:
    training_data = ml.helper.load_training_data()
    ml.helper.save_training_data(*training_data, "expected_fidelity")


def test_create_feature_dict() -> None:
    qc = benchmark_generator.get_benchmark("dj", 1, 3)
    feature_vector = ml.helper.create_feature_dict(qc)
    assert feature_vector is not None

    with pytest.raises(ValueError):
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


def test_predict_device_for_figure_of_merit() -> None:
    qc = benchmark_generator.get_benchmark("ghz", 1, 5)
    assert (
        ml.helper.predict_device_for_figure_of_merit(qc, "expected_fidelity")
        in ml.helper.get_index_to_device_LUT().values()
    )


def test_qcompile() -> None:
    qc = benchmark_generator.get_benchmark("ghz", 1, 5)
    qc_compiled, compilation_information, quantum_device = qcompile(qc)
    assert quantum_device in ml.helper.get_index_to_device_LUT().values()
    assert qc_compiled.layout is not None
    assert len(qc_compiled) > 0


def test_get_path_results() -> None:
    for get_ghz_path_results in (True, False):
        path = ml.helper.get_path_results(ghz_results=get_ghz_path_results)
        assert path.exists()
