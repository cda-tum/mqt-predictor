from __future__ import annotations

import logging
import time
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from mqt.bench.tket_helper import get_rebase
from mqt.predictor import Result, ml, reward, rl
from pytket import OpType
from pytket.architecture import Architecture
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from pytket.passes import (
    FullPeepholeOptimise,
    PlacementPass,
    RoutingPass,
)
from pytket.placement import GraphPlacement
from qiskit import QuantumCircuit, transpile

logger = logging.getLogger("mqt-predictor")


def compute_reward_for_compilation_setup(
    benchmark: str,
    compiler: str,
    figure_of_merit: reward.figure_of_merit = "expected_fidelity",
    device: dict[str, Any] | None = None,
) -> Result | None:
    """Computes the reward for a given benchmark with the respectively selected compiler/figure of merit/device and returns the results as a Result object.

    Args:
        benchmark (str): The path to the benchmark to be compiled.
        compiler (str): The setup to be used for compilation. Either 'mqt-predictor', 'qiskit_o3' or 'tket'.
        figure_of_merit (reward.reward_functions, optional): The figure of merit to be used for compilation. Defaults to "expected_fidelity".
        device (dict[str, Any] | None, optional): The device to be used for compilation. Defaults to None.

    Returns:
        Result | None: Returns a Result object containing the compiled quantum circuit, the compilation information and the name of the device used for compilation. If compilation fails, None is returned.
    """
    if compiler == "mqt-predictor":
        return create_mqtpredictor_result(benchmark, compiler, figure_of_merit)

    qc = QuantumCircuit.from_qasm_file(benchmark)
    if "qiskit" in compiler:
        return create_qiskit_result(benchmark, compiler, qc, device)

    if "tket" in compiler:
        return create_tket_result(benchmark, compiler, qc, device)

    error_msg = "Unknown setup. Use either 'mqt-predictor', 'qiskit' or 'tket'."
    raise ValueError(error_msg)


def create_qiskit_result(
    benchmark: str, compiler: str, qc: QuantumCircuit, device: dict[str, Any] | None = None
) -> Result | None:
    assert device is not None
    if qc.num_qubits > device["max_qubits"]:
        return Result(benchmark, compiler, -1, None, device["name"])
    start_time = time.time()
    try:
        transpiled_qc_qiskit = transpile(
            qc,
            basis_gates=device["native_gates"],
            coupling_map=device["cmap"],
            optimization_level=3,
            seed_transpiler=1,
        )
    except Exception as e:
        logger.warning("Qiskit Transpile Error occurred for: " + benchmark + " " + device["name"] + " " + str(e))
        return Result(benchmark, compiler, -1, None, device["name"])
    duration = time.time() - start_time
    return Result(benchmark, compiler, duration, transpiled_qc_qiskit, device["name"])


def create_tket_result(
    benchmark: str,
    compiler: str,
    qc: QuantumCircuit,
    device: dict[str, Any] | None = None,
) -> Result | None:
    assert device is not None
    if qc.num_qubits > device["max_qubits"]:
        return Result(benchmark, compiler, -1, None, device["name"])
    tket_qc = qiskit_to_tk(qc)
    arch = Architecture(device["cmap"])

    native_rebase = get_rebase(device["name"].split("_")[0])
    assert native_rebase is not None

    start_time = time.time()
    try:
        native_rebase.apply(tket_qc)
        FullPeepholeOptimise(target_2qb_gate=OpType.TK2).apply(tket_qc)
        PlacementPass(GraphPlacement(arch)).apply(tket_qc)
        RoutingPass(arch).apply(tket_qc)
        native_rebase.apply(tket_qc)
        duration = time.time() - start_time
        transpiled_qc_tket = tk_to_qiskit(tket_qc)
    except Exception as e:
        logger.warning("TKET Transpile Error occurred for: " + benchmark + " " + device["name"] + " " + str(e))
        return Result(benchmark, compiler, -1, None, device["name"])

    return Result(benchmark, compiler, duration, transpiled_qc_tket, device["name"])


def create_mqtpredictor_result(benchmark: str, compiler: str, figure_of_merit: reward.figure_of_merit) -> Result | None:
    dev_name = ml.helper.get_predicted_and_suitable_device_name(
        QuantumCircuit.from_qasm_file(benchmark), figure_of_merit
    )
    assert isinstance(dev_name, str)
    dev_index = next((index for index, d in enumerate(rl.helper.get_devices()) if d["name"] == dev_name), None)
    target_filename = benchmark.split("/")[-1].split(".qasm")[0] + "_" + figure_of_merit + "_" + str(dev_index)
    combined_path_filename = ml.helper.get_path_training_circuits_compiled() / (target_filename + ".qasm")
    if Path(combined_path_filename).exists():
        qc = QuantumCircuit.from_qasm_file(combined_path_filename)
        if qc:
            return Result(
                benchmark,
                compiler + "_" + figure_of_merit,
                -1,
                qc,
                dev_name,
            )
    else:
        try:
            qc_compiled = ml.qcompile(QuantumCircuit.from_qasm_file(benchmark), figure_of_merit=figure_of_merit)
            if qc_compiled:
                assert isinstance(qc_compiled, tuple)
                return Result(
                    benchmark,
                    compiler + "_" + figure_of_merit,
                    -1,
                    qc_compiled[0],
                    dev_name,
                )

        except Exception as e:
            logger.warning("mqt-predictor Transpile Error occurred for: " + benchmark + " " + dev_name + " " + str(e))
    return Result(benchmark, compiler, -1, None, dev_name)


def evaluate_all_sample_circuits() -> None:
    """Evaluates all sample circuits and saves the results to a csv file."""
    res_csv = []

    results = Parallel(n_jobs=-1, verbose=3, backend="threading")(
        delayed(evaluate_sample_circuit)(str(file))
        for file in list(ml.helper.get_path_training_circuits().glob("*.qasm"))
    )
    res_csv.append(list(results[0].keys()))
    for res in results:
        res_csv.append(list(res.values()))
    np.savetxt(
        ml.helper.get_path_trained_model() / "res.csv",
        res_csv,
        delimiter=",",
        fmt="%s",
    )


def evaluate_GHZ_circuits() -> None:
    """Evaluates all GHZ circuits and saves the results to a csv file."""
    res_csv = []

    path = Path(str(resources.files("mqt.predictor"))) / "ml" / "training_data" / "ghz"
    results = Parallel(n_jobs=-1, verbose=3, backend="threading")(
        delayed(evaluate_sample_circuit)(str(file)) for file in list(path.glob("*.qasm"))
    )
    res_csv.append(list(results[0].keys()))
    for res in results:
        res_csv.append(list(res.values()))
    np.savetxt(
        ml.helper.get_path_trained_model() / "res_GHZ.csv",
        res_csv,
        delimiter=",",
        fmt="%s",
    )


def evaluate_sample_circuit(file: str) -> dict[str, Any]:
    """Evaluates a given sample circuit and returns the results as a dictionary.

    Args:
        file (str): The path to the sample circuit to be evaluated.

    Returns:
        dict[str, Any]: Returns a dictionary containing the results of the evaluation.
    """
    print("Evaluate file: " + file)
    logger.info("Evaluate file: " + file)

    results = []
    results.append(compute_reward_for_compilation_setup(file, "mqt-predictor", "expected_fidelity"))
    results.append(compute_reward_for_compilation_setup(file, "mqt-predictor", "critical_depth"))

    for _i, dev in enumerate(rl.helper.get_devices()):
        results.append(compute_reward_for_compilation_setup(file, "qiskit_" + dev["name"], device=dev))

    for _i, dev in enumerate(rl.helper.get_devices()):
        results.append(compute_reward_for_compilation_setup(file, "tket_" + dev["name"], device=dev))

    combined_res: dict[str, Any] = {
        "file_path": str(Path(file).stem),
        "benchmark_name": str(Path(file).stem).replace("_", " ").split(" ")[0],
        "num_qubits": str(Path(file).stem).replace("_", " ").split(" ")[-1],
    }

    for res in results:
        if res is None:
            logger.warning("Error occurred for: " + file)
            continue
        assert res is not None
        combined_res.update(res.get_dict())
    return combined_res
