from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, get_args

import numpy as np
from joblib import Parallel, delayed
from mqt.bench.qiskit_helper import get_native_gates
from mqt.bench.utils import get_cmap_from_devicename
from mqt.predictor import Result, ml, rl
from pytket import OpType
from pytket.architecture import Architecture  # type: ignore[attr-defined]
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from pytket.passes import (  # type: ignore[attr-defined]
    FullPeepholeOptimise,
    PlacementPass,
    RoutingPass,
    auto_rebase_pass,
)
from pytket.placement import GraphPlacement  # type: ignore[attr-defined]
from qiskit import QuantumCircuit, transpile

logger = logging.getLogger("mqtpredictor")


def computeRewards(
    benchmark: str,
    used_setup: str,
    opt_objective: rl.helper.reward_functions = "fidelity",
) -> Result | None:
    if used_setup == "MQTPredictor":
        qc = QuantumCircuit.from_qasm_file(benchmark)
        start_time = time.time()
        ml_predictor = ml.Predictor()
        predicted_device_index = ml_predictor.predict(benchmark)
        device_name = rl.helper.get_devices()[predicted_device_index]["name"]
        res = rl.qcompile(qc, device_name=device_name)
        duration = time.time() - start_time

        if res:
            assert type(res) == tuple
            return Result(
                benchmark,
                used_setup + "_" + opt_objective,
                duration,
                res[0],
                rl.helper.get_devices()[predicted_device_index]["name"],
            )
        return None

    if used_setup == "qiskit_o3":
        qc = QuantumCircuit.from_qasm_file(benchmark)
        start_time = time.time()
        transpiled_qc_qiskit = transpile(
            qc,
            basis_gates=get_native_gates("ibm"),
            coupling_map=get_cmap_from_devicename("ibm_washington"),
            optimization_level=3,
            seed_transpiler=1,
        )
        duration = time.time() - start_time

        return Result(benchmark, used_setup, duration, transpiled_qc_qiskit, "ibm_washington")

    if used_setup == "tket":
        qc = QuantumCircuit.from_qasm_file(benchmark)
        tket_qc = qiskit_to_tk(qc)
        arch = Architecture(get_cmap_from_devicename("ibm_washington"))
        ibm_rebase = auto_rebase_pass({OpType.Rz, OpType.SX, OpType.X, OpType.CX, OpType.Measure})

        start_time = time.time()
        ibm_rebase.apply(tket_qc)
        FullPeepholeOptimise(target_2qb_gate=OpType.TK2).apply(tket_qc)
        PlacementPass(GraphPlacement(arch)).apply(tket_qc)
        RoutingPass(arch).apply(tket_qc)
        ibm_rebase.apply(tket_qc)
        duration = time.time() - start_time
        transpiled_qc_tket = tk_to_qiskit(tket_qc)

        return Result(benchmark, used_setup, duration, transpiled_qc_tket, "ibm_washington")

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


def evaluate_sample_circuit(file: str) -> dict[str, Any]:
    print("Evaluate file: " + file)
    logger.info("Evaluate file: " + file)

    # reward_functions = ["fidelity", "critical_depth", "gate_ratio", "mix"]
    get_args(rl.helper.reward_functions)
    results = []
    # for rew in reward_functions:
    print("Calc MQT Predictor Reward")
    results.append(computeRewards(file, "MQTPredictor", "fidelity"))

    print("Calc Qiskit O3 Reward")
    results.append(computeRewards(file, "qiskit_o3"))
    print("Calc tket Reward")
    results.append(computeRewards(file, "tket"))

    combined_res: dict[str, Any] = {
        "benchmark": str(Path(file).stem).replace("_", " ").split(" ")[0],
        "num_qubits": str(Path(file).stem).replace("_", " ").split(" ")[-1],
    }

    for res in results:
        assert res is not None
        combined_res.update(res.get_dict())
    return combined_res
