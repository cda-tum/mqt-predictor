"""This module contains functions for evaluating the performance of the mqt-predictor."""

from __future__ import annotations

import logging
import time
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed
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

from mqt.bench.devices import Device, get_available_device_names, get_available_devices
from mqt.bench.tket_helper import get_rebase
from mqt.predictor import Result, ml, reward

logger = logging.getLogger("mqt-predictor")


def create_qiskit_result(qc: QuantumCircuit, device: Device) -> Result:
    """Creates a Result object for a given benchmark and device using qiskit for compilation.

    Arguments:
        qc: The quantum circuit to be compiled.
        device: The device to be used for compilation. Defaults to None.

    Returns:
        A Result object containing the compiled quantum circuit.
    """
    if qc.num_qubits > device.num_qubits:
        return Result("qiskit_", -1, None, device)
    start_time = time.time()
    try:
        transpiled_qc_qiskit = transpile(
            qc,
            basis_gates=device.basis_gates,
            coupling_map=device.coupling_map,
            optimization_level=3,
            seed_transpiler=1,
        )
    except Exception as e:
        logger.warning("qiskit Transpile Error occurred for: " + device.name + " " + str(e))
        return Result("qiskit_" + device.name, -1, None, device)
    duration = time.time() - start_time
    return Result("qiskit_" + device.name, duration, transpiled_qc_qiskit, device)


def create_tket_result(
    qc: QuantumCircuit,
    device: Device,
) -> Result:
    """Creates a Result object for a given benchmark and device using tket for compilation.

    Arguments:
        qc: The quantum circuit to be compiled.
        device: The device to be used for compilation. Defaults to None.

    Returns:
        A Result object containing the compiled quantum circuit.
    """
    if qc.num_qubits > device.num_qubits:
        return Result("tket_" + device.name, -1, None, device)
    try:
        tket_qc = qiskit_to_tk(qc)
        arch = Architecture(device.coupling_map)

        native_rebase = get_rebase(device.basis_gates)
        assert native_rebase is not None

        start_time = time.time()

        native_rebase.apply(tket_qc)
        FullPeepholeOptimise(target_2qb_gate=OpType.TK2).apply(tket_qc)
        PlacementPass(GraphPlacement(arch)).apply(tket_qc)
        RoutingPass(arch).apply(tket_qc)
        native_rebase.apply(tket_qc)
        duration = time.time() - start_time
        transpiled_qc_tket = tk_to_qiskit(tket_qc)
    except Exception as e:
        logger.warning("tket Transpile Error occurred for: " + device.name + " " + str(e))
        return Result("tket_" + device.name, -1, None, device)

    return Result("tket_" + device.name, duration, transpiled_qc_tket, device)


def create_mqtpredictor_result(qc: QuantumCircuit, figure_of_merit: reward.figure_of_merit, filename: str) -> Result:
    """Creates a Result object for a given benchmark and figure of merit using mqt-predictor for compilation.

    Arguments:
        qc: The quantum circuit to be compiled.
        figure_of_merit: The figure of merit to be used for compilation.
        filename: The path to the benchmark to be compiled.


    Returns:
        A Result object containing the compiled quantum circuit.
    """
    device = ml.helper.predict_device_for_figure_of_merit(qc, figure_of_merit)
    dev_index = get_available_device_names().index(device.name)
    target_filename = filename.split("/")[-1].split(".qasm")[0] + "_" + figure_of_merit + "_" + str(dev_index)
    combined_path_filename = ml.helper.get_path_training_circuits_compiled() / (target_filename + ".qasm")
    if Path(combined_path_filename).exists():
        qc = QuantumCircuit.from_qasm_file(combined_path_filename)
        if qc:
            return Result(
                "mqt-predictor_" + figure_of_merit,
                -1,
                qc,
                device,
            )
    else:
        try:
            qc_compiled = ml.qcompile(qc, figure_of_merit=figure_of_merit)
            if qc_compiled:
                assert isinstance(qc_compiled, tuple)
                return Result(
                    "mqt-predictor_" + figure_of_merit,
                    -1,
                    qc_compiled[0],
                    device,
                )

        except Exception as e:
            logger.warning("mqt-predictor Transpile Error occurred for: " + filename + " " + device.name + " " + str(e))
    return Result("mqt-predictor_" + figure_of_merit, -1, None, device)


def evaluate_all_sample_circuits() -> None:
    """Evaluates all sample circuits and saves the results to a csv file."""
    res_csv = []

    results = Parallel(n_jobs=-1, verbose=3, backend="threading")(
        delayed(evaluate_sample_circuit)(str(file))
        for file in list(ml.helper.get_path_training_circuits().glob("*.qasm"))
    )
    res_csv.append(list(results[0].keys()))
    res_csv.extend([list(res.values()) for res in results])
    np.savetxt(
        ml.helper.get_path_results(),
        res_csv,
        delimiter=",",
        fmt="%s",
    )


def evaluate_ghz_circuits() -> None:
    """Evaluates all GHZ circuits and saves the results to a csv file."""
    res_csv = []

    path = Path(str(resources.files("mqt.predictor"))) / "ml" / "training_data" / "ghz"
    results = Parallel(n_jobs=-1, verbose=3, backend="threading")(
        delayed(evaluate_sample_circuit)(str(file)) for file in list(path.glob("*.qasm"))
    )
    res_csv.append(list(results[0].keys()))
    res_csv.extend([list(res.values()) for res in results])
    np.savetxt(
        ml.helper.get_path_results(ghz_results=True),
        res_csv,
        delimiter=",",
        fmt="%s",
    )


def evaluate_sample_circuit(filename: str) -> dict[str, Any]:
    """Evaluates a given sample circuit and returns the results as a dictionary.

    Arguments:
        filename: The path to the sample circuit to be evaluated.
        devices: The devices to be used for compilation.

    Returns:
        A dictionary containing the results of the evaluation.
    """
    logger.info("Evaluate file: " + filename)

    results: dict[str, Any] = {
        "file_path": str(Path(filename).stem),
        "benchmark_name": str(Path(filename).stem).replace("_", " ").split(" ")[0],
        "num_qubits": str(Path(filename).stem).replace("_", " ").split(" ")[-1],
    }
    qc = QuantumCircuit.from_qasm_file(filename)
    results.update(create_mqtpredictor_result(qc, "expected_fidelity", filename=filename).get_dict())
    results.update(create_mqtpredictor_result(qc, "critical_depth", filename=filename).get_dict())

    for dev in get_available_devices():
        results.update(create_qiskit_result(qc, dev).get_dict())
        results.update(create_tket_result(qc, dev).get_dict())

    return results
