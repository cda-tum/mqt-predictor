from __future__ import annotations

from mqt.bench import benchmark_generator
from mqt.bench.devices import get_available_device_names, get_available_devices
from mqt.predictor import ml, qcompile


def test_load_training_data() -> None:
    assert ml.helper.load_training_data() is not None


def test_create_feature_dict() -> None:
    qc = benchmark_generator.get_benchmark("dj", 1, 3)
    feature_vector = ml.helper.create_feature_dict(qc)
    assert feature_vector is not None


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
    assert ml.helper.predict_device_for_figure_of_merit(qc, "expected_fidelity") in get_available_devices()


def test_qcompile() -> None:
    qc = benchmark_generator.get_benchmark("ghz", 1, 5)
    qc_compiled, compilation_information, quantum_device = qcompile(qc)
    assert quantum_device in get_available_device_names()
    assert qc_compiled.layout is not None
    assert len(qc_compiled) > 0
