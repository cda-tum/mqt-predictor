from __future__ import annotations

import logging
from pathlib import Path

from mqt.predictor import rl
from qiskit import QuantumCircuit
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks

logger = logging.getLogger("mqtpredictor")
PATH_LENGTH = 260


class Predictor:
    def __init__(self, verbose: int = 0):
        if verbose == 1:
            lvl = logging.INFO
        elif verbose == 2:
            lvl = logging.DEBUG
        else:
            lvl = logging.WARNING
        logger.setLevel(lvl)

    def compile_as_predicted(
        self,
        qc: QuantumCircuit | str,
        opt_objective: rl.helper.reward_functions = "fidelity",
        device_name: str = "ibm_washington",
    ) -> tuple[QuantumCircuit, list[str]] | bool:
        if not isinstance(qc, QuantumCircuit):
            if len(qc) < PATH_LENGTH and Path(qc).exists():
                qc = QuantumCircuit.from_qasm_file(qc)
            elif "OPENQASM" in qc:
                qc = QuantumCircuit.from_qasm_str(qc)
        print("read model: ", "model_" + opt_objective + "_" + device_name)
        model = rl.helper.load_model("model_" + opt_objective + "_" + device_name)
        env = rl.PredictorEnv(opt_objective, device_name)
        obs, _ = env.reset(qc)

        used_compilation_passes = []
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action_masks = get_action_masks(env)
            action, _ = model.predict(obs, action_masks=action_masks)
            action = int(action)
            action_item = env.action_set[action]
            used_compilation_passes.append(action_item["name"])
            obs, reward_val, terminated, truncated, info = env.step(action)
            # print("Action taken: ", action_item["name"])
            # print("Gate: ", env.state.count_ops())
            env.state._layout = env.layout

        if env.state.count_ops().get("u"):
            print("Warning: u gates are still present in the circuit")
            print("Error occurred: ", env.error_occured)

        if not env.error_occured:
            return env.state, used_compilation_passes
        return False

    def train_all_models(
        self,
        timesteps: int = 1000,
        reward_functions: list[rl.helper.reward_functions] | None = None,
        model_name: str = "model",
        device_name: str = "ibm_washington",
        verbose: int = 2,
        test: bool = False,
    ) -> None:
        if reward_functions is None:
            reward_functions = ["fidelity"]
        if test:
            n_steps = 100
            progress_bar = False
        else:
            n_steps = 2048
            progress_bar = True

        for rew in reward_functions:
            logger.debug("Start training for: " + rew)
            env = rl.PredictorEnv(reward_function=rew, device_name=device_name)

            model = MaskablePPO(
                MaskableMultiInputActorCriticPolicy,
                env,
                verbose=verbose,
                tensorboard_log="./" + model_name + "_" + rew + "_" + device_name,
                gamma=0.98,
                n_steps=n_steps,
            )
            model.learn(total_timesteps=timesteps, progress_bar=progress_bar)
            model.save(rl.helper.get_path_trained_model() / (model_name + "_" + rew + "_" + device_name))
