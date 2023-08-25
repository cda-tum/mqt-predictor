from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from mqt.bench.tket_helper import get_rebase
from mqt.predictor import Result, ml, qcompile, reward, rl
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


# def evaluate_baseline() -> None:
#     res_csv = []
#
#     results = Parallel(n_jobs=-1, verbose=3, backend="threading")(
#         delayed(eval_baseline_compilation)(str(file))
#         for file in list(rl.helper.get_path_training_circuits().glob("*.qasm"))[:50]
#     )
#     res_csv.append([results[0].get_dict().keys()])
#     for res in results:
#         res_csv.append([res.get_dict().values()])
#     np.savetxt(
#         ml.helper.get_path_trained_model() / "baseline.csv",
#         res_csv,
#         delimiter=",",
#         fmt="%s",
#     )


def computeRewards(
    benchmark: str,
    used_setup: str,
    figure_of_merit: reward.reward_functions = "fidelity",
    device: dict[str, Any] | None = None,
) -> Result | None:
    if used_setup == "MQTPredictor":
        qc = QuantumCircuit.from_qasm_file(benchmark)
        start_time = time.time()
        res = qcompile(qc, figure_of_merit=figure_of_merit)

        duration = time.time() - start_time

        if res:
            assert type(res) == tuple
            return Result(
                benchmark,
                used_setup + "_" + figure_of_merit,
                duration,
                res[0],
                res[2],
            )
        return None

    if "qiskit" in used_setup:
        assert device is not None
        qc = QuantumCircuit.from_qasm_file(benchmark)
        if qc.num_qubits > device["max_qubits"]:
            return Result(benchmark, used_setup, -1, None, device["name"])
        start_time = time.time()
        transpiled_qc_qiskit = transpile(
            qc,
            basis_gates=device["native_gates"],  # get_native_gates("ibm"),
            coupling_map=device["cmap"],  # get_cmap_from_devicename("ibm_washington"),
            optimization_level=3,
            seed_transpiler=1,
        )
        duration = time.time() - start_time

        return Result(benchmark, used_setup, duration, transpiled_qc_qiskit, device["name"])

    if "tket" in used_setup:
        assert device is not None
        qc = QuantumCircuit.from_qasm_file(benchmark)
        if qc.num_qubits > device["max_qubits"]:
            return Result(benchmark, used_setup, -1, None, device["name"])
        tket_qc = qiskit_to_tk(qc)
        arch = Architecture(device["cmap"])
        # arch = Architecture(get_cmap_from_devicename("ibm_washington"))
        # ibm_rebase = auto_rebase_pass({OpType.Rz, OpType.SX, OpType.X, OpType.CX, OpType.Measure})

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
        native_rebase.apply(tket_qc)
        FullPeepholeOptimise(target_2qb_gate=OpType.TK2).apply(tket_qc)
        PlacementPass(GraphPlacement(arch)).apply(tket_qc)
        RoutingPass(arch).apply(tket_qc)
        native_rebase.apply(tket_qc)
        duration = time.time() - start_time
        transpiled_qc_tket = tk_to_qiskit(tket_qc)

        return Result(benchmark, used_setup, duration, transpiled_qc_tket, device["name"])

    error_msg = "Unknown setup. Use either 'RL', 'qiskit_o3' or 'tket'."
    raise ValueError(error_msg)


# def eval_baseline_compilation(
#     benchmark: str,
# ) -> BaselineResult | None:
#     qc = QuantumCircuit.from_qasm_file(benchmark)
#     qiskit_res: list[tuple[float, float, float]] = []
#     tket_res: list[tuple[float, float, float]] = []
#     print(benchmark)
#
#     for _i, device in enumerate(rl.helper.get_devices()):
#         if qc.num_qubits > ["max_qubits"]:
#             qiskit_res.append((-1.0, -1.0, -1.0))
#             tket_res.append((-1.0, -1.0, -1.0))
#             continue
#         print("Start with device: ", device["name"])
#         print("qiskit")
#         start_time = time.time()
#         transpiled_qc_qiskit = transpile(
#             qc,
#             basis_gates=device["native_gates"],
#             coupling_map=device["cmap"],
#             optimization_level=3,
#             seed_transpiler=1,
#         )
#         duration = time.time() - start_time
#         qiskit_fid = reward.expected_fidelity(transpiled_qc_qiskit, device=device["name"])
#         qiskit_crit_dep = reward.crit_depth(transpiled_qc_qiskit)
#         qiskit_res.append((qiskit_fid, qiskit_crit_dep, duration))
#
#         print("tket")
#         qc_tket_copy = qc.copy()
#         tket_qc = qiskit_to_tk(qc_tket_copy)
#         arch = Architecture(device["cmap"])
#         if "ibm" in device["name"]:
#             native_rebase = get_rebase("ibm")
#         elif "oqc" in device["name"]:
#             native_rebase = get_rebase("oqc")
#         elif "rigetti" in device["name"]:
#             native_rebase = get_rebase("rigetti")
#         elif "quantinuum" in device["name"]:
#             native_rebase = get_rebase("quantinuum")
#         elif "ionq" in device["name"]:
#             native_rebase = get_rebase("ionq")
#         else:
#             msg = "Unknown Native Gate-Set"
#             raise RuntimeError(msg)
#
#         start_time = time.time()
#         native_rebase.apply(tket_qc)
#         FullPeepholeOptimise(target_2qb_gate=OpType.TK2).apply(tket_qc)
#         PlacementPass(GraphPlacement(arch)).apply(tket_qc)
#         RoutingPass(arch).apply(tket_qc)
#         native_rebase.apply(tket_qc)
#         duration = time.time() - start_time
#         transpiled_qc_tket = tk_to_qiskit(tket_qc)
#         tket_fid = reward.expected_fidelity(transpiled_qc_tket, device=device["name"])
#         tket_crit_dep = reward.crit_depth(transpiled_qc_tket)
#         tket_res.append((tket_fid, tket_crit_dep, duration))
#
#     return BaselineResult(benchmark, qiskit_res, tket_res)


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


def evaluate_sample_circuit(file: str) -> dict[str, Any]:
    print("Evaluate file: " + file)
    logger.info("Evaluate file: " + file)

    results = []
    # for rew in reward_functions:
    print("Calc MQT Predictor Reward Fid")
    results.append(computeRewards(file, "MQTPredictor", "fidelity"))
    print("Calc MQT Predictor Reward Dep")
    results.append(computeRewards(file, "MQTPredictor", "critical_depth"))

    print("Calc Qiskit O3 Reward")
    for _i, dev in enumerate(rl.helper.get_devices()):
        results.append(computeRewards(file, "qiskit_" + dev["name"], device=dev))

    print("Calc tket Reward")
    for _i, dev in enumerate(rl.helper.get_devices()):
        results.append(computeRewards(file, "tket_" + dev["name"], device=dev))

    combined_res: dict[str, Any] = {
        "benchmark": str(Path(file).stem).replace("_", " ").split(" ")[0],
        "num_qubits": str(Path(file).stem).replace("_", " ").split(" ")[-1],
    }

    for res in results:
        assert res is not None
        combined_res.update(res.get_dict())
    return combined_res
