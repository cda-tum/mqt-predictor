from pathlib import Path

from mqt.bench import benchmark_generator, qiskit_helper
from mqt.predictor import ml, reward


def test_get_width_penalty() -> None:
    assert ml.helper.get_width_penalty() < 0


def test_get_index_to_comppath_LUT() -> None:
    expected = {
        0: "ibm_washington",
        1: "ibm_montreal",
        2: "oqc_lucy",
        3: "rigetti_aspen_m2",
        4: "ionq_harmony",
        5: "ionq_aria1",
        6: "quantinuum_h2",
    }
    assert ml.helper.get_index_to_comppath_LUT() == expected


def test_get_compilation_pipeline() -> None:
    expected = {
        "devices": {
            "ibm": [("ibm_washington", 127), ("ibm_montreal", 27)],
            "rigetti": [("rigetti_aspen_m2", 80)],
            "ionq": [("ionq_harmony", 11), ("ionq_aria1", 25)],
            "oqc": [("oqc_lucy", 8)],
            "quantinuum": [("quantinuum_h2", 32)],
        },
        "compiler": {
            "qiskit": {"optimization_level": [0, 1, 2, 3]},
            "tket": {"lineplacement": [False, True]},
        },
    }
    assert ml.helper.get_compilation_pipeline() == expected


def test_load_training_data() -> None:
    assert ml.helper.load_training_data() is not None


def test_calc_eval_score_for_qc() -> None:
    qc = benchmark_generator.get_benchmark("dj", 1, 3)
    compilation_pipeline = ml.helper.get_compilation_pipeline()

    filename_qasm = "eval_test.qasm"
    for gate_set_name, devices in compilation_pipeline["devices"].items():
        for device_name, max_qubits in devices:
            for compiler, settings in compilation_pipeline["compiler"].items():
                if "qiskit" in compiler:
                    for opt_level in settings["optimization_level"]:
                        if max_qubits >= qc.num_qubits:
                            qiskit_helper.get_mapped_level(
                                qc,
                                gate_set_name,
                                qc.num_qubits,
                                device_name,
                                opt_level,
                                False,
                                False,
                                ".",
                                "eval_test",
                            )
                            score = reward.expected_fidelity(filename_qasm, device=device_name)
                            assert score >= 0 and score <= 1 or score == ml.helper.get_width_penalty()

    if Path(filename_qasm).is_file():
        Path(filename_qasm).unlink()
