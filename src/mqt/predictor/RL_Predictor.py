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

from mqt.predictor import RL_utils, utils
from mqt.predictor.PhaseOrdererEnv import PhaseOrdererEnv


class RL_Predictor:
    def compile(self, qc, opt_objective="fidelity"):
        if not isinstance(qc, QuantumCircuit):
            if len(qc) < 260 and Path(qc).exists():
                qc = QuantumCircuit.from_qasm_file(qc)
            elif "OPENQASM" in qc:
                qc = QuantumCircuit.from_qasm_str(qc)

        model = MaskablePPO.load(
            RL_utils.get_path_trained_model_RL() / ("model_" + opt_objective)
        )
        env = PhaseOrdererEnv(opt_objective)
        obs, _ = env.reset(qc)
        while True:
            action_masks = get_action_masks(env)
            action, _states = model.predict(obs, action_masks=action_masks)
            action = int(action)
            action_item = env.action_set.get(action)
            if action_item in RL_utils.get_actions_devices():
                device = action_item["name"]
            obs, reward, done, trunc, info = env.step(action)
            if done:
                return env.state, device

    def evaluate_sample_circuit_using_RL(self, file):
        print(file)

        reward_functions = ["parallelism", "fidelity", "critical_depth"]
        for rew in reward_functions:
            model = MaskablePPO.load(
                RL_utils.get_path_trained_model_RL() / ("model_" + rew)
            )

            env = PhaseOrdererEnv(rew)
            obs, _ = env.reset(file)
            qc = env.state
            start_time = time.time()
            while True:
                action_masks = get_action_masks(env)
                action, _states = model.predict(obs, action_masks=action_masks)
                action = int(action)
                obs, reward, done, trunc, info = env.step(action)
                if done:
                    duration = time.time() - start_time
                    if rew == "fidelity":
                        RL_fid = np.round(
                            utils.reward_expected_fidelity(env.state, env.device), 2
                        )
                        RL_fid_time = np.round(duration, 2)
                        RL_fid_crit_depth = np.round(
                            utils.reward_crit_depth(env.state), 2
                        )
                        RL_fid_parallelism = np.round(
                            utils.reward_parallelism(env.state), 2
                        )
                    elif rew == "parallelism":
                        RL_parallelism = np.round(
                            utils.reward_parallelism(env.state), 2
                        )
                        RL_parallelism_time = np.round(duration, 2)
                        RL_parallelism_fid = np.round(
                            utils.reward_expected_fidelity(env.state, env.device), 2
                        )
                        RL_parallelism_crit_depth = np.round(
                            utils.reward_crit_depth(env.state), 2
                        )
                    elif rew == "critical_depth":
                        RL_crit_depth = np.round(utils.reward_crit_depth(env.state), 2)
                        RL_crit_depth_time = np.round(duration, 2)
                        RL_crit_depth_fid = np.round(
                            utils.reward_expected_fidelity(env.state, env.device), 2
                        )
                        RL_crit_depth_parallelism = np.round(
                            utils.reward_parallelism(env.state), 2
                        )
                    break

        start_time = time.time()
        transpiled_qc_qiskit = transpile(
            qc,
            basis_gates=RL_utils.get_ibm_native_gates(),
            coupling_map=RL_utils.get_cmap_from_devicename("ibm_washington"),
            optimization_level=3,
            seed_transpiler=1,
        )
        duration = time.time() - start_time
        qiskit_o3_fid = np.round(
            utils.reward_expected_fidelity(transpiled_qc_qiskit, "ibm_washington"), 2
        )
        qiskit_o3_crit_depth = utils.reward_crit_depth(transpiled_qc_qiskit)
        qiskit_o3_parallel = utils.reward_parallelism(transpiled_qc_qiskit)
        qiskit_o3_time = np.round(duration, 2)

        tket_qc = qiskit_to_tk(qc)
        arch = architecture.Architecture(
            RL_utils.get_cmap_from_devicename("ibm_washington")
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
            utils.reward_expected_fidelity(transpiled_qc_tket, "ibm_washington"), 2
        )
        tket_crit_depth = utils.reward_crit_depth(transpiled_qc_tket)
        tket_parallelism = utils.reward_parallelism(transpiled_qc_tket)
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

    def eval_all_sample_circuits_using_RL(self):
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
            for file in list(RL_utils.get_path_training_circuits_RL().glob("*.qasm"))
        )
        for res in results:
            res_csv.append(res)
        # print(results)
        np.savetxt("res.csv", res_csv, delimiter=",", fmt="%s")

    def instantiate_RL_models(self, timesteps, fid, dep, par, model_name="training"):

        utils.init_all_config_files()
        self.train_all_RL_models(
            timestep=timesteps,
            verbose=2,
            fid=fid,
            dep=dep,
            par=par,
            model_name=model_name,
        )

    def train_all_RL_models(
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
            print("fidelity added")
        if dep:
            reward_functions.append("critical_depth")
            print("critical_depth added")
        if par:
            reward_functions.append("parallelism")
            print("parallelism added")

        for rew in reward_functions:
            print("Start training for: ", rew)
            env = PhaseOrdererEnv(reward_function=rew)

            model = MaskablePPO(
                MaskableActorCriticPolicy,
                env,
                verbose=verbose,
                tensorboard_log="./" + model_name + "_" + rew,
                gamma=0.98,
            )
            model.learn(total_timesteps=timestep, progress_bar=True)
            model.save(RL_utils.get_path_trained_model_RL() / (model_name + "_" + rew))
