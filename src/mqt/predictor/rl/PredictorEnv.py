from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import quantum_generative_modeling
from bqskit.ext import bqskit_to_qiskit, qiskit_to_bqskit
from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete
from pytket.circuit import Qubit
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import ApplyLayout, CheckMap, GatesInBasis
from qiskit.transpiler.runningpassmanager import TranspileLayout

from mqt.predictor import reward, rl

logger = logging.getLogger("mqt-predictor")


class PredictorEnv(Env):  # type: ignore[misc]
    """Predictor environment for reinforcement learning."""

    def __init__(
        self,
        reward_function: reward.figure_of_merit = "expected_fidelity",
        device_name: str = "ibm_washington",
        num_qubits: int = 4,
    ):
        logger.info("Init env: " + reward_function)
        logger.info("Device: " + device_name)
        logger.info("Num qubits: " + str(num_qubits))

        self.action_set = {}
        self.actions_synthesis_indices = []
        self.actions_layout_indices = []
        self.actions_routing_indices = []
        self.actions_mapping_indices = []
        self.actions_opt_indices = []
        self.actions_final_optimization_indices = []
        self.used_actions: list[str] = []

        self.device = rl.helper.get_device(device_name)

        index = 0

        for elem in rl.helper.get_actions_synthesis():
            self.action_set[index] = elem
            self.actions_synthesis_indices.append(index)
            index += 1
        for elem in rl.helper.get_actions_layout():
            self.action_set[index] = elem
            self.actions_layout_indices.append(index)
            index += 1
        for elem in rl.helper.get_actions_routing():
            self.action_set[index] = elem
            self.actions_routing_indices.append(index)
            index += 1
        for elem in rl.helper.get_actions_opt():
            self.action_set[index] = elem
            self.actions_opt_indices.append(index)
            index += 1
        for elem in rl.helper.get_actions_mapping():
            self.action_set[index] = elem
            self.actions_mapping_indices.append(index)
            index += 1
        for elem in rl.helper.get_actions_final_optimization():
            self.action_set[index] = elem
            self.actions_final_optimization_indices.append(index)
            index += 1

        self.action_set[index] = rl.helper.get_action_terminate()
        self.action_terminate_index = index

        self.reward_function = reward_function
        self.action_space = Discrete(len(self.action_set.keys()))
        self.num_steps = 0
        self.layout: TranspileLayout | None = None

        spaces = {
            "num_qubits": Discrete(128),
            "depth": Discrete(1000000),
            "program_communication": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "critical_depth": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "entanglement_ratio": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "parallelism": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "liveness": Box(low=0, high=1, shape=(1,), dtype=np.float32),
        }
        self.observation_space = Dict(spaces)
        self.filename = ""

        if reward_function == "KL":
            self.state: QuantumCircuit = quantum_generative_modeling.generate_circuit(n_qubits=num_qubits)
        else:
            self.state, self.filename = rl.helper.get_state_sample()

        from qiskit import transpile

        baseline_compiled_qc = transpile(
            self.state, coupling_map=self.device["cmap"], basis_gates=self.device["native_gates"], optimization_level=3
        )
        print("Baseline Gate Counts: ", baseline_compiled_qc.count_ops())
        self.max_cx_count = baseline_compiled_qc.count_ops().get("cx", 0) * 1.05
        self.initial_uncompiled_circuit = self.state.copy()
        self.has_parametrized_gates = len(self.state.parameters) > 0
        self.num_qubits_uncompiled_circuit = self.state.num_qubits
        self.best_KL: float = 0.0
        self.best_compilation_sequence: list[str] = []

        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if reward_function == "KL":
            with Path(
                f"evaluations/results_application_aware_compilation/mqt_predictor_{self.num_qubits_uncompiled_circuit}_qubits_"
                + str(self.timestamp)
                + ".txt"
            ).open(mode="w+") as file:
                file.write("KL values and number of gates for\n")
                file.write("Device: " + str(self.device["name"]) + "\n")
                file.write("Reward function: " + str(self.reward_function) + "\n")
                file.write("Number of qubits: " + str(self.num_qubits_uncompiled_circuit) + "\n")

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[Any, Any]]:
        """Executes the given action and returns the new state, the reward, whether the episode is done, whether the episode is truncated and additional information."""
        self.used_actions.append(str(self.action_set[action].get("name")))
        altered_qc = self.apply_action(action)
        if (
            not altered_qc
            or sum(altered_qc.count_ops().values()) > sum(self.initial_uncompiled_circuit.count_ops().values()) * 15
        ):
            return (
                rl.helper.create_feature_dict(self.state),
                0,
                True,
                False,
                {},
            )

        self.state = altered_qc
        self.num_steps += 1
        self.valid_actions = self.determine_valid_actions_for_state()

        if len(self.valid_actions) == 0:
            msg = "No valid actions left."
            raise RuntimeError(msg)

        if action == self.action_terminate_index:
            reward_val = self.calculate_reward()
            done = True
        else:
            reward_val = 0
            done = False

        # in case the Qiskit.QuantumCircuit has unitary or u gates in it, decompose them (because otherwise qiskit will throw an error when applying the BasisTranslator
        if self.state.count_ops().get("unitary"):
            self.state = self.state.decompose(gates_to_decompose="unitary")

        self.state._layout = self.layout  # noqa: SLF001
        obs = rl.helper.create_feature_dict(self.state)
        return obs, reward_val, done, False, {}

    def calculate_reward(self) -> Any:
        """Calculates and returns the reward for the current state."""
        if self.reward_function == "expected_fidelity":
            return reward.expected_fidelity(self.state, self.device["name"])
        if self.reward_function == "critical_depth":
            return reward.crit_depth(self.state)
        if self.reward_function == "KL":
            new_KL_value, evaluation_data = reward.KL(
                self.state,
                sum(self.initial_uncompiled_circuit.count_ops().values()),
                self.max_cx_count,
                self.num_qubits_uncompiled_circuit,
                self.device["name"],
            )
            print("Current best value: " + str(self.best_KL))
            print("New value: " + str(new_KL_value))

            if new_KL_value > 0.0:
                with Path(
                    f"evaluations/results_application_aware_compilation/mqt_predictor_{self.num_qubits_uncompiled_circuit}_qubits_"
                    + str(self.timestamp)
                    + ".txt"
                ).open(mode="a") as file:
                    file.write(str(new_KL_value) + " " + str(self.state.count_ops()) + "\n")

            if new_KL_value > self.best_KL:
                self.best_KL = new_KL_value
                self.best_compilation_sequence = self.used_actions
                with Path(
                    f"evaluations/results_application_aware_compilation/mqt_predictor_{self.num_qubits_uncompiled_circuit}_qubits_"
                    + str(self.timestamp)
                    + ".txt"
                ).open(mode="a") as file:
                    file.write("evaluation_data:" + str(evaluation_data) + "\n")
                    file.write(str(self.best_compilation_sequence) + "\n")

            return new_KL_value
        return None  # type: ignore[unreachable]

    def render(self) -> None:
        """Renders the current state."""
        print(self.state.draw())

    def reset(
        self,
        qc: Path | str | QuantumCircuit | None = None,
        seed: int | None = None,
        options: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> tuple[QuantumCircuit, dict[str, Any]]:
        """Resets the environment to the given state or a random state.

        Args:
            qc (Path | str | QuantumCircuit | None, optional): The quantum circuit to be compiled or the path to a qasm file containing the quantum circuit. Defaults to None.
            seed (int | None, optional): The seed to be used for the random number generator. Defaults to None.
            options (dict[str, Any] | None, optional): Additional options. Defaults to None.

        Returns:
            tuple[QuantumCircuit, dict[str, Any]]: The initial state and additional information.
        """
        super().reset(seed=seed)

        self.action_space = Discrete(len(self.action_set.keys()))
        self.num_steps = 0
        self.used_actions = []
        self.layout = None
        self.valid_actions = self.actions_opt_indices + self.actions_synthesis_indices
        self.error_occured = False

        if self.reward_function == "KL":
            self.state = self.initial_uncompiled_circuit
        else:
            if isinstance(qc, QuantumCircuit):
                self.state = qc
            elif qc:
                self.state = QuantumCircuit.from_qasm_file(str(qc))
            else:
                self.state, self.filename = rl.helper.get_state_sample()

        return rl.helper.create_feature_dict(self.state), {}

    def action_masks(self) -> list[bool]:
        """Returns a list of valid actions for the current state."""
        action_mask = [action in self.valid_actions for action in self.action_set]

        if self.layout is not None:
            action_mask = [
                action_mask[i] and self.action_set[i].get("origin") != "tket" for i in range(len(action_mask))
            ]
        if self.has_parametrized_gates or self.layout is not None:
            # remove all actions that are from "origin"=="bqskit" because they are not supported for parametrized gates
            # or after layout since using BQSKit after a layout is set may result in an error
            action_mask = [
                action_mask[i] and self.action_set[i].get("origin") != "bqskit" for i in range(len(action_mask))
            ]
        return action_mask

    def apply_action(self, action_index: int) -> QuantumCircuit | None:
        """Applies the given action to the current state and returns the altered state."""
        if action_index in self.action_set:
            action = self.action_set[action_index]
            if action["name"] == "terminate":
                return self.state
            if action_index in self.actions_opt_indices:
                transpile_pass = action["transpile_pass"]
            else:
                transpile_pass = action["transpile_pass"](self.device)

            if action["origin"] == "qiskit":
                try:
                    if action["name"] == "QiskitO3":
                        pm = PassManager()
                        pm.append(
                            action["transpile_pass"](
                                self.device["native_gates"],
                                CouplingMap(self.device["cmap"]) if self.layout is not None else None,
                            ),
                            do_while=action["do_while"],
                        )
                    else:
                        pm = PassManager(transpile_pass)
                    altered_qc = pm.run(self.state)
                except Exception:
                    logger.exception(
                        "Error in executing Qiskit transpile pass for {action} at step {i} for {filename}".format(
                            action=action["name"], i=self.num_steps, filename=self.filename
                        )
                    )

                    self.error_occured = True
                    return None
                if (
                    action_index
                    in self.actions_layout_indices
                    + self.actions_mapping_indices
                    + self.actions_final_optimization_indices
                ):
                    if action["name"] == "VF2Layout":
                        if pm.property_set["layout"]:
                            initial_layout = pm.property_set["layout"]
                            original_qubit_indices = pm.property_set["original_qubit_indices"]
                            final_layout = pm.property_set["final_layout"]
                            postprocessing_action = rl.helper.get_layout_postprocessing_qiskit_pass()(self.device)
                            pm = PassManager(postprocessing_action)
                            pm.property_set["layout"] = initial_layout
                            pm.property_set["original_qubit_indices"] = original_qubit_indices
                            pm.property_set["final_layout"] = final_layout
                            altered_qc = pm.run(altered_qc)
                    elif action["name"] == "VF2PostLayout":
                        assert pm.property_set["VF2PostLayout_stop_reason"] is not None
                        post_layout = pm.property_set["post_layout"]
                        if post_layout:
                            pm = PassManager(ApplyLayout())
                            assert self.layout is not None
                            pm.property_set["layout"] = self.layout.initial_layout
                            pm.property_set["original_qubit_indices"] = self.layout.input_qubit_mapping
                            pm.property_set["final_layout"] = self.layout.final_layout
                            pm.property_set["post_layout"] = post_layout
                            altered_qc = pm.run(altered_qc)
                    else:
                        assert pm.property_set["layout"]

                    if pm.property_set["layout"]:
                        self.layout = TranspileLayout(
                            initial_layout=pm.property_set["layout"],
                            input_qubit_mapping=pm.property_set["original_qubit_indices"],
                            final_layout=pm.property_set["final_layout"],
                            _output_qubit_list=altered_qc.qubits,
                            _input_qubit_count=self.num_qubits_uncompiled_circuit,
                        )
                elif action_index in self.actions_routing_indices:
                    assert self.layout is not None
                    self.layout.final_layout = pm.property_set["final_layout"]

            elif action["origin"] == "tket":
                try:
                    tket_qc = qiskit_to_tk(self.state)
                    for elem in transpile_pass:
                        elem.apply(tket_qc)
                    qbs = tket_qc.qubits
                    qubit_map = {qbs[i]: Qubit("q", i) for i in range(len(qbs))}
                    tket_qc.rename_units(qubit_map)  # type: ignore[arg-type]
                    altered_qc = tk_to_qiskit(tket_qc)
                    if action_index in self.actions_routing_indices:
                        assert self.layout is not None
                        self.layout.final_layout = rl.helper.final_layout_pytket_to_qiskit(tket_qc, altered_qc)

                except Exception:
                    logger.exception(
                        "Error in executing TKET transpile  pass for {action} at step {i} for {filename}".format(
                            action=action["name"], i=self.num_steps, filename=self.filename
                        )
                    )
                    self.error_occured = True
                    return None

            elif action["origin"] == "bqskit":
                try:
                    bqskit_qc = qiskit_to_bqskit(self.state)
                    if action_index in self.actions_opt_indices + self.actions_synthesis_indices:
                        bqskit_compiled_qc = transpile_pass(bqskit_qc)
                        altered_qc = bqskit_to_qiskit(bqskit_compiled_qc)
                    elif action_index in self.actions_mapping_indices:
                        bqskit_compiled_qc, initial_layout, final_layout = transpile_pass(bqskit_qc)
                        altered_qc = bqskit_to_qiskit(bqskit_compiled_qc)
                        layout = rl.helper.final_layout_bqskit_to_qiskit(
                            initial_layout, final_layout, altered_qc, self.state
                        )
                        self.layout = layout
                except Exception:
                    logger.exception(
                        "Error in executing BQSKit transpile  pass for {action} at step {i} for {filename}".format(
                            action=action["name"], i=self.num_steps, filename=self.filename
                        )
                    )
                    self.error_occured = True
                    return None

            else:
                error_msg = f"Origin {action['origin']} not supported."
                raise ValueError(error_msg)

        else:
            error_msg = f"Action {action_index} not supported."
            raise ValueError(error_msg)

        return altered_qc

    def determine_valid_actions_for_state(self) -> list[int]:
        """Determines and returns the valid actions for the current state."""
        check_nat_gates = GatesInBasis(basis_gates=self.device["native_gates"])
        check_nat_gates(self.state)
        only_nat_gates = check_nat_gates.property_set["all_gates_in_basis"]

        if not only_nat_gates:
            actions = self.actions_synthesis_indices + self.actions_opt_indices
            if self.layout is not None:
                actions += self.actions_routing_indices
            return actions

        check_mapping = CheckMap(coupling_map=CouplingMap(self.device["cmap"]))
        check_mapping(self.state)
        mapped = check_mapping.property_set["is_swap_mapped"]

        if mapped and self.layout is not None:
            return [self.action_terminate_index, *self.actions_opt_indices, *self.actions_final_optimization_indices]

        if self.layout is not None:
            return self.actions_routing_indices

        # No layout applied yet
        return self.actions_mapping_indices + self.actions_layout_indices + self.actions_opt_indices
