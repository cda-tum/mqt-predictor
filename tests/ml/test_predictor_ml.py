from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from mqt.bench import benchmark_generator
from mqt.predictor import ml


@patch("matplotlib.pyplot.show")
def test_predict(mock_show: Any) -> None:  # noqa: ARG001
    path = ml.helper.get_path_trained_model() / "trained_clf.joblib"
    assert path.is_file()
    qc = benchmark_generator.get_benchmark("dj", 1, 8)
    predictor = ml.Predictor()
    prediction = predictor.predict(qc)
    assert 0 <= prediction < len(ml.helper.get_index_to_compilation_path_dict())
    prediction = predictor.predict(qc.qasm())
    assert 0 <= prediction < len(ml.helper.get_index_to_compilation_path_dict())
    with pytest.raises(ValueError, match="Invalid input for 'qc' parameter"):
        predictor.predict(Path("Error Test"))

    predictor.clf = None
    filename = "test_qasm.qasm"
    qc.qasm(filename=filename)
    prediction = predictor.predict(Path(filename))
    Path(filename).unlink()
    assert 0 <= prediction < len(ml.helper.get_index_to_compilation_path_dict())


@pytest.mark.parametrize("comp_path", list(range(len(ml.helper.get_index_to_compilation_path_dict()))))
def test_compilation_paths(comp_path: int) -> None:  # noqa: ARG001
    qc = benchmark_generator.get_benchmark("dj", 1, 2)
    res, compile_info = ml.qcompile(qc)
    assert res
    assert compile_info

    tmp_filename = "test.qasm"
    qc.qasm(filename=tmp_filename)
    res, compile_info = ml.qcompile(Path(tmp_filename))
    assert res
    assert compile_info

    if Path(tmp_filename).exists():
        Path(tmp_filename).unlink()


def test_compile_all_circuits_for_qc() -> None:
    qc = benchmark_generator.get_benchmark("dj", 1, 2)
    tmp_filename = "test.qasm"
    qc.qasm(filename=tmp_filename)
    predictor = ml.Predictor()
    assert predictor.compile_all_circuits_for_qc(
        filename=tmp_filename,
        source_path=Path.cwd(),
    )
    if Path(tmp_filename).exists():
        Path(tmp_filename).unlink()


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
    source_path = Path.cwd()
    target_path = Path("test_compiled_circuits")
    if not target_path.exists():
        target_path.mkdir()

    qc = benchmark_generator.get_benchmark("dj", 1, 3)
    qasm_path = Path("compiled_test.qasm")
    qc.qasm(filename=str(qasm_path))
    predictor.generate_compiled_circuits(source_path, target_path)
    assert any(file.suffix == ".qasm" for file in target_path.iterdir())

    res = predictor.generate_training_sample(
        qasm_path,
        path_uncompiled_circuit=source_path,
        path_compiled_circuits=target_path,
    )
    assert not isinstance(res, bool)
    training_sample, circuit_name, scores = res
    assert circuit_name
    assert scores

    (
        training_data,
        name_list,
        scores_list,
    ) = predictor.generate_trainingdata_from_qasm_files(source_path, target_path)
    assert training_data
    assert name_list
    assert scores_list

    if target_path.exists():
        for file in target_path.iterdir():
            file.unlink()
        target_path.rmdir()

    if qasm_path.exists():
        qasm_path.unlink()
