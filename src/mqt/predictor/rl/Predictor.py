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
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks

from mqt.predictor import reward, rl


class Predictor:
    def compile_as_predicted(self, qc, opt_objective="fidelity"):
        if not isinstance(qc, QuantumCircuit):
            if len(qc) < 260 and Path(qc).exists():
                qc = QuantumCircuit.from_qasm_file(qc)
            elif "OPENQASM" in qc:
                qc = QuantumCircuit.from_qasm_str(qc)

        model = MaskablePPO.load(
            rl.helper.get_path_trained_model() / ("model_" + opt_objective)
        )
        env = rl.PhaseOrdererEnv(opt_objective)
        obs = env.reset(qc)
        while True:
            action_masks = get_action_masks(env)
            action, _states = model.predict(obs, action_masks=action_masks)
            action = int(action)
            action_item = env.action_set.get(action)
            if action_item in rl.helper.get_actions_devices():
                device = action_item["name"]
            obs, reward_val, done, info = env.step(action)
            if done:
                return env.state, device

    def evaluate_sample_circuit_using(self, file):
        print(file)

        reward_functions = ["parallelism", "fidelity", "critical_depth"]
        for rew in reward_functions:
            model = MaskablePPO.load(
                rl.helper.get_path_trained_model() / ("model_" + rew)
            )

            env = rl.PhaseOrdererEnv(rew)
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
                        RL_fid = np.round(
                            reward.expected_fidelity(env.state, env.device), 2
                        )
                        RL_fid_time = np.round(duration, 2)
                        RL_fid_crit_depth = np.round(reward.crit_depth(env.state), 2)
                        RL_fid_parallelism = np.round(reward.parallelism(env.state), 2)
                    elif rew == "parallelism":
                        RL_parallelism = np.round(reward.parallelism(env.state), 2)
                        RL_parallelism_time = np.round(duration, 2)
                        RL_parallelism_fid = np.round(
                            reward.expected_fidelity(env.state, env.device), 2
                        )
                        RL_parallelism_crit_depth = np.round(
                            reward.crit_depth(env.state), 2
                        )
                    elif rew == "critical_depth":
                        RL_crit_depth = np.round(reward.crit_depth(env.state), 2)
                        RL_crit_depth_time = np.round(duration, 2)
                        RL_crit_depth_fid = np.round(
                            reward.expected_fidelity(env.state, env.device), 2
                        )
                        RL_crit_depth_parallelism = np.round(
                            reward.parallelism(env.state), 2
                        )
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
        qiskit_o3_fid = np.round(
            reward.expected_fidelity(transpiled_qc_qiskit, "ibm_washington"), 2
        )
        qiskit_o3_crit_depth = reward.crit_depth(transpiled_qc_qiskit)
        qiskit_o3_parallel = reward.parallelism(transpiled_qc_qiskit)
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

        tket_fid = np.round(
            reward.expected_fidelity(transpiled_qc_tket, "ibm_washington"), 2
        )
        tket_crit_depth = reward.crit_depth(transpiled_qc_tket)
        tket_parallelism = reward.parallelism(transpiled_qc_tket)
        tket_time = np.round(duration, 2)

        return (
            str(file).split("/")[-1].split(".")[0].replace("_", " ").split(" ")[0],
            str(file).split("/")[-1].split(".")[0].replace("_", " ").split(" ")[-1],
            qiskit_o3_fid,
            tket_fid,
            RL_fid,
            qiskit_o3_parallel,
            tket_parallelism,
            RL_parallelism,
            qiskit_o3_crit_depth,
            tket_crit_depth,
            RL_crit_depth,
            qiskit_o3_time,
            tket_time,
            RL_fid_time,
            RL_parallelism_time,
            RL_crit_depth_time,
            RL_fid_crit_depth,
            RL_fid_parallelism,
            RL_crit_depth_fid,
            RL_crit_depth_parallelism,
            RL_parallelism_fid,
            RL_parallelism_crit_depth,
        )

    def eval_all_sample_circuits(self):
        res_csv = []
        res_csv.append(
            (
                "Benchmark",
                "Qubits",
                "Qiskit O3 Fid",
                "TKET Fid",
                "RL Fid",
                "Qiskit O3 Parallelism",
                "TKET Parallelism",
                "RL Parallelism",
                "Qiskit O3 Crit Depth",
                "TKET Crit Depth",
                "RL Crit Depth",
                "Qiskit O3 Time",
                "TKET Time",
                "RL Fid Time",
                "RL Parallelism Time",
                "RL Crit Depth Time",
                "RL Fid Crit Depth",
                "RL Fid Parallelism",
                "RL Crit Depth Fid",
                "RL Crit Depth Parallelism",
                "RL Parallelism Fid",
                "RL Parallelism Crit Depth",
            )
        )

        results = Parallel(n_jobs=-1, verbose=3, backend="threading")(
            delayed(self.evaluate_sample_circuit)(file)
            for file in list(rl.helper.get_path_training_circuits().glob("*.qasm"))
        )
        for res in results:
            res_csv.append(res)
        # print(results)
        np.savetxt(
            rl.helper.get_path_trained_model() / "res.csv",
            res_csv,
            delimiter=",",
            fmt="%s",
        )

    def instantiate_models(self, timesteps, fid, dep, par, model_name="training"):
        self.train_all_models(
            timestep=timesteps,
            verbose=2,
            fid=fid,
            dep=dep,
            par=par,
            model_name=model_name,
        )

    def train_all_models(
        self,
        timestep=1000,
        verbose=2,
        fid=False,
        dep=False,
        par=False,
        model_name="model",
    ):
        reward_functions = []  # ["fidelity", "depth", "two_qubit_count"]
        if fid:
            reward_functions.append("fidelity")
        if dep:
            reward_functions.append("critical_depth")
        if par:
            reward_functions.append("parallelism")

        for rew in reward_functions:
            print("Start training for: ", rew)
            env = rl.PhaseOrdererEnv(reward_function=rew)

            model = MaskablePPO(
                MaskableActorCriticPolicy,
                env,
                verbose=verbose,
                tensorboard_log="./" + model_name + "_" + rew,
                gamma=0.98,
            )
            model.learn(total_timesteps=timestep, progress_bar=False)
            model.save(rl.helper.get_path_trained_model() / (model_name + "_" + rew))
