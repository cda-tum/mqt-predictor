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
from pytket.architecture import Architecture  # type: ignore[attr-defined]
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from pytket.passes import (  # type: ignore[attr-defined]
    FullPeepholeOptimise,
    PlacementPass,
    RoutingPass,
)
from pytket.placement import GraphPlacement  # type: ignore[attr-defined]
from qiskit import QuantumCircuit, transpile

logger = logging.getLogger("mqtpredictor")


def computeRewards(  # noqa: PLR0911
    benchmark: str,
    used_setup: str,
    figure_of_merit: reward.reward_functions = "fidelity",
    device: dict[str, Any] | None = None,
) -> Result | None:
    if used_setup == "MQTPredictor":
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
                    used_setup + "_" + figure_of_merit,
                    -1,
                    qc,
                    dev_name,
                )
        # else:
        #     try:
        #         qc_compiled = ml.qcompile(QuantumCircuit.from_qasm_file(benchmark), figure_of_merit=figure_of_merit)
        #         if qc_compiled:
        #             assert isinstance(qc_compiled, tuple)
        #             return Result(
        #                 benchmark,
        #                 used_setup + "_" + figure_of_merit,
        #                 -1,
        #                 qc_compiled[0],
        #                 dev_name,
        #             )

        # except Exception as e:
        #     print("Error occurred for: ", benchmark, dev_name, e)
        #     return Result(benchmark, used_setup, -1, None, dev_name)

        return Result(benchmark, used_setup, -1, None, dev_name)

    qc = QuantumCircuit.from_qasm_file(benchmark)
    if "qiskit" in used_setup:
        assert device is not None
        if qc.num_qubits > device["max_qubits"]:
            return Result(benchmark, used_setup, -1, None, device["name"])
        start_time = time.time()
        try:
            transpiled_qc_qiskit = transpile(
                qc,
                basis_gates=device["native_gates"],  # get_native_gates("ibm"),
                coupling_map=device["cmap"],  # get_cmap_from_devicename("ibm_washington"),
                optimization_level=3,
                seed_transpiler=1,
            )
        except Exception as e:
            print("Qiskit Transpile Error for: ", benchmark, device["name"], e)
            return Result(benchmark, used_setup, -1, None, device["name"])

        duration = time.time() - start_time

        return Result(benchmark, used_setup, duration, transpiled_qc_qiskit, device["name"])

    if "tket" in used_setup:
        assert device is not None
        if qc.num_qubits > device["max_qubits"]:
            return Result(benchmark, used_setup, -1, None, device["name"])
        tket_qc = qiskit_to_tk(qc)
        arch = Architecture(device["cmap"])

        if "ibm" in device["name"]:
            native_rebase = get_rebase("ibm")
        elif "oqc" in device["name"]:
            native_rebase = get_rebase("oqc")
        elif "rigetti" in device["name"]:
            native_rebase = get_rebase("rigetti")
        elif "quantinuum" in device["name"]:
            native_rebase = get_rebase("quantinuum")
        elif "ionq" in device["name"]:
            native_rebase = get_rebase("ionq")
        else:
            msg = "Unknown Native Gate-Set"
            raise RuntimeError(msg)

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
            print("TKET Transpile Error for: ", benchmark, device["name"], e)
            return Result(benchmark, used_setup, -1, None, device["name"])

        return Result(benchmark, used_setup, duration, transpiled_qc_tket, device["name"])

    error_msg = "Unknown setup. Use either 'RL', 'qiskit_o3' or 'tket'."
    raise ValueError(error_msg)


def evaluate_all_sample_circuits() -> None:
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
    res_csv = []

    path = Path(str(resources.files("mqt.predictor"))) / "ml" / "training_data" / "GHZ"
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
    print("Evaluate file: " + file)
    logger.info("Evaluate file: " + file)

    results = []
    results.append(computeRewards(file, "MQTPredictor", "fidelity"))
    results.append(computeRewards(file, "MQTPredictor", "critical_depth"))

    for _i, dev in enumerate(rl.helper.get_devices()):
        results.append(computeRewards(file, "qiskit_" + dev["name"], device=dev))

    for _i, dev in enumerate(rl.helper.get_devices()):
        results.append(computeRewards(file, "tket_" + dev["name"], device=dev))

    combined_res: dict[str, Any] = {
        "file_path": str(Path(file).stem),
        "benchmark_name": str(Path(file).stem).replace("_", " ").split(" ")[0],
        "num_qubits": str(Path(file).stem).replace("_", " ").split(" ")[-1],
    }

    for res in results:
        if res is None:
            print("Error occurred for: ", file)
            continue
        assert res is not None
        combined_res.update(res.get_dict())
    return combined_res