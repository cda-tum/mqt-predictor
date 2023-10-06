from mqt.bench import benchmark_generator
from mqt.predictor import ml


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
    path = ml.helper.get_path_trained_model()
    assert path.exists()


def test_qcompile() -> None:
    qc = benchmark_generator.get_benchmark("ghz", 1, 5)
    assert ml.helper.qcompile(qc) is not None
