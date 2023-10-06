from pathlib import Path

import pytest
from mqt.bench import benchmark_generator
from mqt.predictor import ml, reward


def test_predict() -> None:
    path = ml.helper.get_path_trained_model(figure_of_merit="expected_fidelity")
    assert path.is_file()
    filename = "test_qasm.qasm"
    figure_of_merit: reward.figure_of_merit = "expected_fidelity"
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
    predictor.compile_all_circuits_for_dev_and_fom(
        device_name="ibm_washington",
        timeout=10,
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
    assert len(training_data)>0
    assert len(name_list)>0
    assert len(scores_list)>0

    if target_path.exists():
        for file in target_path.iterdir():
            file.unlink()
        target_path.rmdir()

    if qasm_path.exists():
        qasm_path.unlink()
