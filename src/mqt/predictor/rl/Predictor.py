from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Literal

import numpy as np
from joblib import Parallel, delayed
from pytket import OpType, architecture
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from pytket.passes import (
    FullPeepholeOptimise,
    PlacementPass,
    RoutingPass,
    auto_rebase_pass,
)
from pytket.placement import GraphPlacement
from qiskit import QuantumCircuit, transpile
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks

from mqt.predictor import rl


class Predictor:
    def __init__(self, verbose=0):
        self.logger = logging.getLogger("mqtpredictor")
        self.verbose = verbose
        if verbose == 1:
            self.logger.setLevel(logging.INFO)
        elif verbose == 2:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.WARNING)

    def compile_as_predicted(
        self, qc: QuantumCircuit | str, opt_objective: str = "fidelity"
    ):
        if not isinstance(qc, QuantumCircuit):
            if len(qc) < 260 and Path(qc).exists():
                qc = QuantumCircuit.from_qasm_file(qc)
            elif "OPENQASM" in qc:
                qc = QuantumCircuit.from_qasm_str(qc)

        model = rl.helper.load_model("model_" + opt_objective)
        env = rl.PredictorEnv(opt_objective)
        obs = env.reset(qc)

        used_compilation_passes = []
        done = False
        while not done:
            action_masks = get_action_masks(env)
            action, _states = model.predict(obs, action_masks=action_masks)
            action = int(action)
            action_item = env.action_set.get(action)
            used_compilation_passes.append(action_item["name"])
            obs, reward_val, done, info = env.step(action)

        return env.state, used_compilation_passes

    def evaluate_sample_circuit(self, file):
        self.logger.info("Evaluate file: " + str(file))

        reward_functions = ["fidelity", "critical_depth", "gate_ratio", "mix"]
        results = []
        for rew in reward_functions:
            results.append(self.computeRewards(file, "RL", rew))

        results.append(self.computeRewards(file, "qiskit_o3"))
        results.append(self.computeRewards(file, "tket"))

        combined_res = {
            "benchmark": str(Path(file).stem).replace("_", " ").split(" ")[0],
            "num_qubits": str(Path(file).stem).replace("_", " ").split(" ")[-1],
        }

        for res in results:
            combined_res.update(res.get_dict())
        return combined_res

    def evaluate_all_sample_circuits(self):
        res_csv = []

        results = Parallel(n_jobs=-1, verbose=3, backend="threading")(
            delayed(self.evaluate_sample_circuit)(file)
            for file in list(rl.helper.get_path_training_circuits().glob("*.qasm"))
        )
        res_csv.append(list(results[0].keys()))
        for res in results:
            res_csv.append(list(res.values()))
        np.savetxt(
            rl.helper.get_path_trained_model() / "res.csv",
            res_csv,
            delimiter=",",
            fmt="%s",
        )

    Reward = Literal["fidelity", "critical_depth", "mix", "gate_ratio"]

    def train_all_models(
        self,
        timesteps: int = 1000,
        reward_functions: [Reward] = None,
        model_name: str = "model",
        verbose: int = 2,
    ):
        if reward_functions is None:
            reward_functions = ["fidelity"]
        if "test" in model_name:
            n_steps = 100
            progress_bar = False
        else:
            n_steps = 2048
            progress_bar = True

        for rew in reward_functions:
            self.logger.debug("Start training for: " + rew)
            env = rl.PredictorEnv(reward_function=rew)

            model = MaskablePPO(
                MaskableMultiInputActorCriticPolicy,
                env,
                verbose=verbose,
                tensorboard_log="./" + model_name + "_" + rew,
                gamma=0.95,
                n_steps=n_steps,
            )
            model.learn(total_timesteps=timesteps, progress_bar=progress_bar)
            model.save(rl.helper.get_path_trained_model() / (model_name + "_" + rew))

    def computeRewards(
        self, benchmark: str, used_setup: str, reward_function: Reward = "fidelity"
    ) -> rl.Result:
        if used_setup == "RL":
            model = rl.helper.load_model("model_" + reward_function)
            env = rl.PredictorEnv(reward_function)
            obs = env.reset(benchmark)
            start_time = time.time()
            done = False
            while not done:
                action_masks = get_action_masks(env)
                action, _states = model.predict(obs, action_masks=action_masks)
                action = int(action)
                obs, reward_val, done, info = env.step(action)

            duration = time.time() - start_time

            return rl.Result(
                benchmark,
                used_setup + "_" + reward_function,
                duration,
                env.state,
                env.device,
            )

        if used_setup == "qiskit_o3":
            qc = QuantumCircuit.from_qasm_file(benchmark)
            start_time = time.time()
            transpiled_qc_qiskit = transpile(
                qc,
                basis_gates=rl.helper.get_ibm_native_gates(),
                coupling_map=rl.helper.get_cmap_from_devicename("ibm_washington"),
                optimization_level=3,
                seed_transpiler=1,
            )
            duration = time.time() - start_time

            return rl.Result(
                benchmark, used_setup, duration, transpiled_qc_qiskit, "ibm_washington"
            )

        if used_setup == "tket":
            qc = QuantumCircuit.from_qasm_file(benchmark)
            tket_qc = qiskit_to_tk(qc)
            arch = architecture.Architecture(
                rl.helper.get_cmap_from_devicename("ibm_washington")
            )
            ibm_rebase = auto_rebase_pass(
                {OpType.Rz, OpType.SX, OpType.X, OpType.CX, OpType.Measure}
            )

            start_time = time.time()
            ibm_rebase.apply(tket_qc)
            FullPeepholeOptimise(target_2qb_gate=OpType.TK2).apply(tket_qc)
            PlacementPass(GraphPlacement(arch)).apply(tket_qc)
            RoutingPass(arch).apply(tket_qc)
            ibm_rebase.apply(tket_qc)
            duration = time.time() - start_time
            transpiled_qc_tket = tk_to_qiskit(tket_qc)

            return rl.Result(
                benchmark, used_setup, duration, transpiled_qc_tket, "ibm_washington"
            )

        raise ValueError("Unknown setup. Use either 'RL', 'qiskit_o3' or 'tket'.")
