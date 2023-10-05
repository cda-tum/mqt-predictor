from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mqt.predictor import reward, rl
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks

if TYPE_CHECKING:
    from qiskit import QuantumCircuit

logger = logging.getLogger("mqt-predictor")
PATH_LENGTH = 260


class Predictor:
    """The Predictor class is used to compile a given quantum circuit to a device optimizing for the given figure of merit."""

    def __init__(
        self, figure_of_merit: reward.figure_of_merit | None = None, device_name: str | None = None, verbose: int = 0
    ):
        if verbose == 1:
            lvl = logging.INFO
        elif verbose == 2:
            lvl = logging.DEBUG
        else:
            lvl = logging.WARNING
        logger.setLevel(lvl)

        if figure_of_merit is not None and device_name is not None:
            self.model = rl.helper.load_model("model_" + figure_of_merit + "_" + device_name)
            self.env = rl.PredictorEnv(figure_of_merit, device_name)

    def compile_as_predicted(
        self,
        qc: QuantumCircuit,
    ) -> tuple[QuantumCircuit, list[str]]:
        """Compiles a given quantum circuit such that the expected fidelity is maximized by using the respectively trained optimized compiler.

        Args:
            qc (QuantumCircuit | str): The quantum circuit to be compiled or the path to a qasm file containing the quantum circuit.

        Returns:
            tuple[QuantumCircuit, list[str]] | bool: Returns a tuple containing the compiled quantum circuit and the compilation information. If compilation fails, False is returned.
        """

        assert self.model is not None
        assert self.env is not None
        obs, _ = self.env.reset(qc)

        used_compilation_passes = []
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action_masks = get_action_masks(self.env)
            action, _ = self.model.predict(obs, action_masks=action_masks)
            action = int(action)
            action_item = self.env.action_set[action]
            used_compilation_passes.append(action_item["name"])
            obs, reward_val, terminated, truncated, info = self.env.step(action)
            self.env.state._layout = self.env.layout

        if self.env.state.count_ops().get("u"):
            print("Warning: u gates are still present in the circuit")
            print("Error occurred: ", self.env.error_occured)

        if not self.env.error_occured:
            return self.env.state, used_compilation_passes

        msg = "Error occurred during compilation."
        raise Exception(msg)

    def train_all_models(
        self,
        timesteps: int = 1000,
        reward_functions: list[reward.figure_of_merit] | None = None,
        model_name: str = "model",
        device_name: str = "ibm_washington",
        verbose: int = 2,
        test: bool = False,
    ) -> None:
        """Trains all models for the given reward functions and device.

        Args:
            timesteps (int, optional): The number of timesteps to train the model. Defaults to 1000.
            reward_functions (list[reward.reward_functions] | None, optional): The reward functions to train the model for. Defaults to None.
            model_name (str, optional): The name of the model. Defaults to "model".
            device_name (str, optional): The name of the device. Defaults to "ibm_washington".
            verbose (int, optional): The verbosity level. Defaults to 2.
            test (bool, optional): Whether to train the model for testing purposes. Defaults to False.
        """

        if reward_functions is None:
            reward_functions = ["expected_fidelity"]
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
