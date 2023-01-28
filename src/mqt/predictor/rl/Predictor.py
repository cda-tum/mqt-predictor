from __future__ import annotations

import time
from pathlib import Path

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

from mqt.predictor import reward, rl


class Predictor:
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
        while True:
            action_masks = get_action_masks(env)
            action, _states = model.predict(obs, action_masks=action_masks)
            action = int(action)
            action_item = env.action_set.get(action)
            used_compilation_passes.append(action_item["name"])
            obs, reward_val, done, info = env.step(action)
            if done:
                return env.state, used_compilation_passes

    def evaluate_sample_circuit(self, file):
        print("Evaluate file:", file)

        reward_functions = ["fidelity", "critical_depth", "gates", "mix"]
        for rew in reward_functions:
            model = rl.helper.load_model("model_" + rew)
            env = rl.PredictorEnv(rew)
            obs = env.reset(file)

            qc = env.state
            start_time = time.time()
            while True:
                action_masks = get_action_masks(env)
                action, _states = model.predict(obs, action_masks=action_masks)
                action = int(action)
                obs, reward_val, done, info = env.step(action)

                if done:
                    duration = time.time() - start_time
                    if rew == "fidelity":
                        RL_fid = reward.expected_fidelity(env.state, env.device)
                        RL_fid_time = np.round(duration, 2)
                        RL_fid_crit_depth = reward.crit_depth(env.state)
                        RL_fid_gate_ratio = reward.gate_ratio(env.state)
                    elif rew == "gates":
                        RL_gate_ratio = reward.gate_ratio(env.state)
                        RL_gate_ratio_time = np.round(duration, 2)
                        RL_gate_ratio_fid = reward.expected_fidelity(
                            env.state, env.device
                        )
                        RL_gate_ratio_crit_depth = reward.crit_depth(env.state)
                    elif rew == "critical_depth":
                        RL_crit_depth = reward.crit_depth(env.state)
                        RL_crit_depth_time = np.round(duration, 2)
                        RL_crit_depth_fid = reward.expected_fidelity(
                            env.state, env.device
                        )
                        RL_crit_depth_gate_ratio = reward.gate_ratio(env.state)
                    elif rew == "mix":
                        RL_mix = reward.mix(env.state, env.device)
                    break

        start_time = time.time()
        transpiled_qc_qiskit = transpile(
            qc,
            basis_gates=rl.helper.get_ibm_native_gates(),
            coupling_map=rl.helper.get_cmap_from_devicename("ibm_washington"),
            optimization_level=3,
            seed_transpiler=1,
        )
        duration = time.time() - start_time
        qiskit_o3_fid = reward.expected_fidelity(transpiled_qc_qiskit, "ibm_washington")
        qiskit_o3_crit_depth = reward.crit_depth(transpiled_qc_qiskit)
        qiskit_o3_gate_ratio = reward.gate_ratio(transpiled_qc_qiskit)
        qiskit_o3_mix = reward.mix(transpiled_qc_qiskit, "ibm_washington")
        qiskit_o3_time = np.round(duration, 2)

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

        tket_fid = reward.expected_fidelity(transpiled_qc_tket, "ibm_washington")
        tket_crit_depth = reward.crit_depth(transpiled_qc_tket)
        tket_gate_ratio = reward.gate_ratio(transpiled_qc_tket)
        tket_mix = reward.mix(transpiled_qc_tket, "ibm_washington")
        tket_time = np.round(duration, 2)

        res = (
            str(file).split("/")[-1].split(".")[0].replace("_", " ").split(" ")[0],
            str(file).split("/")[-1].split(".")[0].replace("_", " ").split(" ")[-1],
            qiskit_o3_fid,
            tket_fid,
            RL_fid,
            qiskit_o3_gate_ratio,
            tket_gate_ratio,
            RL_gate_ratio,
            qiskit_o3_crit_depth,
            tket_crit_depth,
            RL_crit_depth,
            qiskit_o3_mix,
            tket_mix,
            RL_mix,
            qiskit_o3_time,
            tket_time,
            RL_fid_time,
            RL_gate_ratio_time,
            RL_crit_depth_time,
            RL_fid_crit_depth,
            RL_fid_gate_ratio,
            RL_crit_depth_fid,
            RL_crit_depth_gate_ratio,
            RL_gate_ratio_fid,
            RL_gate_ratio_crit_depth,
        )
        return res

    def evaluate_all_sample_circuits(self):
        res_csv = []
        res_csv.append(
            (
                "Benchmark",
                "Qubits",
                "Qiskit O3 Fid",
                "TKET Fid",
                "RL Fid",
                "Qiskit O3 gate_ratio",
                "TKET gate_ratio",
                "RL gate_ratio",
                "Qiskit O3 Crit Depth",
                "TKET Crit Depth",
                "RL Crit Depth",
                "Qiskit O3 mix",
                "TKET mix",
                "RL mix",
                "Qiskit O3 Time",
                "TKET Time",
                "RL Fid Time",
                "RL gate_ratio Time",
                "RL Crit Depth Time",
                "RL Fid Crit Depth",
                "RL Fid gate_ratio",
                "RL Crit Depth Fid",
                "RL Crit Depth gate_ratio",
                "RL gate_ratio Fid",
                "RL gate_ratio Crit Depth",
            )
        )

        results = Parallel(n_jobs=-1, verbose=3, backend="threading")(
            delayed(self.evaluate_sample_circuit)(file)
            for file in list(rl.helper.get_path_training_circuits().glob("*.qasm"))
        )
        for res in results:
            res_csv.append(res)
        np.savetxt(
            rl.helper.get_path_trained_model() / "res.csv",
            res_csv,
            delimiter=",",
            fmt="%s",
        )

    def train_all_models(
        self,
        timesteps=1000,
        verbose=2,
        fid=False,
        dep=False,
        mix=False,
        gates=False,
        model_name="model",
    ):
        reward_functions = []  # ["fidelity", "depth", "two_qubit_count"]
        if fid:
            reward_functions.append("fidelity")
        if dep:
            reward_functions.append("critical_depth")
        if mix:
            reward_functions.append("mix")
        if gates:
            reward_functions.append("gates")

        if "test" in model_name:
            n_steps = 100
            progress_bar = False
        else:
            n_steps = 2048
            progress_bar = True

        for rew in reward_functions:
            print("Start training for: ", rew)
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
