from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import get_args

from joblib import Parallel, delayed
from mqt.predictor.devices import IBMProvider
from mqt.predictor.rl import PredictorEnv, Result, ResultDict, RewardFunction, Setup
from mqt.predictor.rl.helper import (
    get_path_trained_model,
    get_path_training_circuits,
    load_model,
)
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
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks

logger = logging.getLogger("mqtpredictor")


class Predictor:
    def __init__(self, verbose: int = 0):
        if verbose == 1:
            lvl = logging.INFO
        elif verbose == 2:  # noqa: PLR2004
            lvl = logging.DEBUG
        else:
            lvl = logging.WARNING
        logger.setLevel(lvl)

    @staticmethod
    def compile_as_predicted(
        qc: QuantumCircuit | Path, opt_objective: RewardFunction = "fidelity"
    ) -> tuple[QuantumCircuit, list[str]]:
        if not isinstance(qc, QuantumCircuit):
            qc = QuantumCircuit.from_qasm_file(str(qc))

        model = load_model("model_" + opt_objective)
        env = PredictorEnv(opt_objective)
        obs = env.reset(qc)

        used_compilation_passes = []
        done = False
        while not done:
            action_masks = get_action_masks(env)
            action, _states = model.predict(obs, action_masks=action_masks)
            action_index = int(action)
            action_item = env.action_dict[action_index]
            used_compilation_passes.append(action_item["name"])
            obs, reward_val, done, _ = env.step(action_index)

        return env.state, used_compilation_passes

    def evaluate_sample_circuit(self, file: Path) -> list[ResultDict]:
        logger.info("Evaluate file: " + str(file))

        reward_functions = get_args(RewardFunction)
        results = [self.compute_rewards(file, "RL", rew).get_dict() for rew in reward_functions]
        results.append(self.compute_rewards(file, "qiskit").get_dict())
        results.append(self.compute_rewards(file, "tket").get_dict())
        return results

    def evaluate_all_sample_circuits(self) -> None:
        collective_results: list[list[ResultDict]] = Parallel(n_jobs=-1, verbose=3, backend="threading")(
            delayed(self.evaluate_sample_circuit)(file) for file in list(get_path_training_circuits().glob("*.qasm"))
        )
        assert len(collective_results) > 0
        assert len(collective_results[0]) > 0
        results = collective_results[0]
        # generate headers for csv file
        header: list[str] = ["benchmark", "num_qubits"]
        for result in results:
            identifier = str(result["setup"])
            if result["reward_function"] is not None:
                identifier += "_" + result["reward_function"]
            for key in result:
                if key not in ["benchmark", "num_qubits", "setup", "reward_function"]:
                    header.append(identifier + "_" + key)
        # generate rows for csv file
        rows: list[list[str]] = []
        for results in collective_results:
            assert len(results) > 0
            row = [results[0]["benchmark"], str(results[0]["num_qubits"])]
            for result in results:
                for key, value in result.items():
                    if key not in ["benchmark", "num_qubits", "setup", "reward_function"]:
                        row.append(str(value))
            rows.append(row)
        with Path(get_path_trained_model() / "res.csv").open("w") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

    @staticmethod
    def train_all_models(
        timesteps: int = 1000,
        reward_functions: list[RewardFunction] | None = None,
        model_name: str = "model",
        verbose: int = 2,
    ) -> None:
        if reward_functions is None:
            reward_functions = ["fidelity"]
        if "test" in model_name:
            n_steps = 100
            progress_bar = False
        else:
            n_steps = 2048
            progress_bar = True

        for rew in reward_functions:
            logger.debug("Start training for: " + rew)
            env = PredictorEnv(reward_function=rew)

            model = MaskablePPO(
                MaskableMultiInputActorCriticPolicy,
                env,
                verbose=verbose,
                tensorboard_log="./" + model_name + "_" + rew,
                gamma=0.95,
                n_steps=n_steps,
            )
            model.learn(total_timesteps=timesteps, progress_bar=progress_bar)
            model.save(get_path_trained_model() / (model_name + "_" + rew))

    @staticmethod
    def compute_rewards(
        benchmark: Path,
        setup: Setup,
        reward_function: RewardFunction = "fidelity",
    ) -> Result:
        if setup == "RL":
            model = load_model("model_" + reward_function)
            env = PredictorEnv(reward_function)
            obs = env.reset(benchmark)
            start_time = time.time()
            done = False
            while not done:
                action_masks = get_action_masks(env)
                action, _states = model.predict(obs, action_masks=action_masks)
                obs, reward_val, done, _ = env.step(int(action))

            runtime = time.time() - start_time

            assert env.device is not None
            return Result(
                benchmark=benchmark,
                runtime=runtime,
                qc=env.state,
                device=env.device,
                setup=setup,
                reward_function=reward_function,
            )

        device = IBMProvider.get_device("washington", sanitize_device=True)

        if setup == "qiskit":
            qc = QuantumCircuit.from_qasm_file(str(benchmark))
            start_time = time.time()
            transpiled_qc_qiskit = transpile(
                qc,
                basis_gates=device.basis_gates,
                coupling_map=[[a, b] for a, b in device.coupling_map],
                optimization_level=3,
                seed_transpiler=1,
            )
            runtime = time.time() - start_time

            return Result(
                benchmark=benchmark,
                runtime=runtime,
                qc=transpiled_qc_qiskit,
                device=device,
                setup=setup,
            )

        if setup == "tket":
            qc = QuantumCircuit.from_qasm_file(str(benchmark))
            tket_qc = qiskit_to_tk(qc)
            arch = Architecture(device.coupling_map)
            ibm_rebase = auto_rebase_pass({OpType.Rz, OpType.SX, OpType.X, OpType.CX, OpType.Measure})

            start_time = time.time()
            ibm_rebase.apply(tket_qc)
            FullPeepholeOptimise(target_2qb_gate=OpType.TK2).apply(tket_qc)
            PlacementPass(GraphPlacement(arch)).apply(tket_qc)
            RoutingPass(arch).apply(tket_qc)
            ibm_rebase.apply(tket_qc)
            runtime = time.time() - start_time
            transpiled_qc_tket = tk_to_qiskit(tket_qc)

            return Result(
                benchmark=benchmark,
                runtime=runtime,
                qc=transpiled_qc_tket,
                device=device,
                setup=setup,
            )

        msg = f"Setup {setup} is not supported. Please use one of {get_args(Setup)}"  # type: ignore[unreachable]
        raise ValueError(msg)
