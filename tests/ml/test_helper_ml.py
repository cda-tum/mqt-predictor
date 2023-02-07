from pathlib import Path
from typing import cast

from mqt.bench import benchmark_generator
from mqt.bench.utils import qiskit_helper
from mqt.predictor import ml, reward
from mqt.predictor.ml import QiskitOptions


def test_get_width_penalty() -> None:
    assert ml.helper.get_width_penalty() < 0


def test_load_training_data() -> None:
    assert ml.helper.load_training_data() is not None


def test_calc_eval_score_for_qc() -> None:
    qc = benchmark_generator.get_benchmark("dj", 1, 3)
    compilation_pipeline = ml.helper.get_compilation_pipeline()

    filename_qasm = "eval_test.qasm"
    for provider_name, devices in compilation_pipeline["devices"].items():
        for device in devices:
            for configuration in compilation_pipeline["compiler"]:
                for compiler, settings in configuration.items():
                    if compiler == "qiskit" and qc.num_qubits <= device.num_qubits:
                        qiskit_helper.get_mapped_level(
                            qc,
                            provider_name,
                            qc.num_qubits,
                            device.name,
                            cast(QiskitOptions, settings)["optimization_level"],
                            False,
                            False,
                            ".",
                            "eval_test",
                        )
                        score = reward.expected_fidelity(Path(filename_qasm), device=device)
                        assert 0 <= score <= 1 or score == ml.helper.get_width_penalty()

    if Path(filename_qasm).is_file():
        Path(filename_qasm).unlink()
