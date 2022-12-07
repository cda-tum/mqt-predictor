from pathlib import Path

from mqt.bench import benchmark_generator
from mqt.bench.utils import qiskit_helper

from mqt.predictor import ml, reward


def test_get_width_penalty():
    assert ml.helper.get_width_penalty() < 0


def test_get_index_to_comppath_LUT():
    expected = {
        0: ("ibm", "ibm_washington", "qiskit", 0),
        1: ("ibm", "ibm_washington", "qiskit", 1),
        2: ("ibm", "ibm_washington", "qiskit", 2),
        3: ("ibm", "ibm_washington", "qiskit", 3),
        4: ("ibm", "ibm_washington", "tket", False),
        5: ("ibm", "ibm_washington", "tket", True),
        6: ("ibm", "ibm_montreal", "qiskit", 0),
        7: ("ibm", "ibm_montreal", "qiskit", 1),
        8: ("ibm", "ibm_montreal", "qiskit", 2),
        9: ("ibm", "ibm_montreal", "qiskit", 3),
        10: ("ibm", "ibm_montreal", "tket", False),
        11: ("ibm", "ibm_montreal", "tket", True),
        12: ("rigetti", "rigetti_aspen_m2", "qiskit", 0),
        13: ("rigetti", "rigetti_aspen_m2", "qiskit", 1),
        14: ("rigetti", "rigetti_aspen_m2", "qiskit", 2),
        15: ("rigetti", "rigetti_aspen_m2", "qiskit", 3),
        16: ("rigetti", "rigetti_aspen_m2", "tket", False),
        17: ("rigetti", "rigetti_aspen_m2", "tket", True),
        18: ("ionq", "ionq11", "qiskit", 0),
        19: ("ionq", "ionq11", "qiskit", 1),
        20: ("ionq", "ionq11", "qiskit", 2),
        21: ("ionq", "ionq11", "qiskit", 3),
        22: ("ionq", "ionq11", "tket", False),
        23: ("ionq", "ionq11", "tket", True),
        24: ("oqc", "oqc_lucy", "qiskit", 0),
        25: ("oqc", "oqc_lucy", "qiskit", 1),
        26: ("oqc", "oqc_lucy", "qiskit", 2),
        27: ("oqc", "oqc_lucy", "qiskit", 3),
        28: ("oqc", "oqc_lucy", "tket", False),
        29: ("oqc", "oqc_lucy", "tket", True),
    }
    assert ml.helper.get_index_to_comppath_LUT() == expected


def test_get_compilation_pipeline():
    expected = {
        "devices": {
            "ibm": [("ibm_washington", 127), ("ibm_montreal", 27)],
            "rigetti": [("rigetti_aspen_m2", 80)],
            "ionq": [("ionq11", 11)],
            "oqc": [("oqc_lucy", 8)],
        },
        "compiler": {
            "qiskit": {"optimization_level": [0, 1, 2, 3]},
            "tket": {"lineplacement": [False, True]},
        },
    }
    assert ml.helper.get_compilation_pipeline() == expected


def test_load_training_data():
    assert ml.helper.load_training_data() is not None


def test_calc_eval_score_for_qc():
    qc = benchmark_generator.get_benchmark("dj", 1, 3)
    compilation_pipeline = ml.helper.get_compilation_pipeline()

    filename_qasm = "eval_test.qasm"
    for gate_set_name, devices in compilation_pipeline.get("devices").items():
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
                            score = reward.expected_fidelity(
                                filename_qasm, device=device_name
                            )
                            assert (
                                score >= 0
                                and score <= 1
                                or score == ml.helper.get_width_penalty()
                            )

    if Path(filename_qasm).is_file():
        Path(filename_qasm).unlink()
