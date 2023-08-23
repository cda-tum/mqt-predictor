from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete
from mqt.predictor import reward, rl
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import CheckMap, GatesInBasis
from qiskit.transpiler.runningpassmanager import TranspileLayout

logger = logging.getLogger("mqtpredictor")


class PredictorEnv(Env):  # type: ignore[misc]
    def __init__(self, reward_function: reward.reward_functions = "fidelity", device_name: str = "ibm_washington"):
        logger.info("Init env: " + reward_function)

        self.action_set = {}
        self.actions_synthesis_indices = []
        self.actions_layout_indices = []
        self.actions_routing_indices = []
        self.actions_mapping_indices = []
        self.actions_opt_indices = []
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

        self.action_set[index] = rl.helper.get_action_terminate()
        self.action_terminate_index = index

        self.reward_function = reward_function
        self.action_space = Discrete(len(self.action_set.keys()))
        self.num_steps = 0

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

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[Any, Any]]:
        self.used_actions.append(str(self.action_set[action].get("name")))
        altered_qc = self.apply_action(action)
        if not altered_qc:
            return (
                rl.helper.create_feature_dict(self.state),
                0,
                True,
                False,
                {},
            )

        self.state: QuantumCircuit = altered_qc
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

        obs = rl.helper.create_feature_dict(self.state)
        return obs, reward_val, done, False, {}

    def calculate_reward(self) -> Any:
        if self.reward_function == "fidelity":
            return reward.expected_fidelity(self.state, self.device["name"])
        if self.reward_function == "critical_depth":
            return reward.crit_depth(self.state)
        if self.reward_function == "mix":
            return reward.mix(self.state, self.device["name"])
        if self.reward_function == "gate_ratio":
            return reward.gate_ratio(self.state)
        error_msg = f"Reward function {self.reward_function} not supported."  # type: ignore[unreachable]
        raise ValueError(error_msg)

    def render(self) -> None:
        print(self.state.draw())

    def reset(
        self,
        qc: Path | str | QuantumCircuit | None = None,
        seed: int | None = None,
        options: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> tuple[QuantumCircuit, dict[str, Any]]:
        super().reset(seed=seed)
        if isinstance(qc, QuantumCircuit):
            self.state = qc
        elif qc:
            self.state = QuantumCircuit.from_qasm_file(str(qc))
        else:
            self.state, self.filename = rl.helper.get_state_sample(self.device["max_qubits"])

        self.action_space = Discrete(len(self.action_set.keys()))
        self.num_steps = 0
        self.used_actions = []

        self.layout = None

        self.valid_actions = self.actions_opt_indices + self.actions_synthesis_indices

        self.error_occured = False
        return rl.helper.create_feature_dict(self.state), {}

    def action_masks(self) -> list[bool]:
        return [action in self.valid_actions for action in self.action_set]

    def apply_action(self, action_index: int) -> QuantumCircuit:
        if action_index in self.action_set:
            action = self.action_set[action_index]
            if action["name"] == "terminate":
                return self.state
            if (
                action_index
                in self.actions_layout_indices + self.actions_routing_indices + self.actions_mapping_indices
            ):
                transpile_pass = action["transpile_pass"](self.device["cmap"])
            elif action_index in self.actions_synthesis_indices:
                transpile_pass = action["transpile_pass"](self.device["native_gates"])
            else:
                transpile_pass = action["transpile_pass"]
            if action["origin"] == "qiskit":
                pm = PassManager(transpile_pass)
                try:
                    altered_qc = pm.run(self.state)
                except Exception as e:
                    print(
                        "Error in executing Qiskit transpile pass: "
                        + action["name"]
                        + ", steps: "
                        + str(self.num_steps)
                        + ", "
                        + str(self.filename)
                        + ", "
                        + str(e)
                    )
                    print(self.used_actions)
                    self.error_occured = True
                    return None
                if action_index in self.actions_layout_indices + self.actions_mapping_indices:
                    assert pm.property_set["layout"]
                    self.layout = TranspileLayout(
                        initial_layout=pm.property_set["layout"],
                        input_qubit_mapping=pm.property_set["original_qubit_indices"],
                        final_layout=pm.property_set["final_layout"],
                    )

            elif action["origin"] == "tket":
                try:
                    tket_qc = qiskit_to_tk(self.state)
                    for elem in transpile_pass:
                        elem.apply(tket_qc)
                    altered_qc = tk_to_qiskit(tket_qc)
                except Exception as e:
                    print(
                        "Error in executing TKET transpile pass: "
                        + action["name"]
                        + ", steps: "
                        + str(self.num_steps)
                        + ", "
                        + self.state.name
                        + ", "
                        + str(self.filename)
                        + ", "
                        + str(e)
                    )
                    self.error_occured = True
                    return None
            # elif action["origin"] == "mqt":
            #     print(self.state.count_ops())
            #     altered_qc = transpile_pass(self.state)[0]
            # elif action["origin"] == "bqskit":
            #     bqskit_qc = qiskit_to_bqskit(self.state)
            #     compiled_bqskit_qc = transpile_pass(bqskit_qc)
            #     altered_qc = bqskit_to_qiskit(compiled_bqskit_qc)

            else:
                error_msg = f"Origin {action['origin']} not supported."
                raise ValueError(error_msg)

        else:
            error_msg = f"Action {action_index} not supported."
            raise ValueError(error_msg)

        return altered_qc

    def determine_valid_actions_for_state(self) -> list[int]:
        check_nat_gates = GatesInBasis(basis_gates=self.device["native_gates"])
        check_nat_gates(self.state)
        only_nat_gates = check_nat_gates.property_set["all_gates_in_basis"]

        if not only_nat_gates:
            return self.actions_synthesis_indices + self.actions_opt_indices

        check_mapping = CheckMap(coupling_map=CouplingMap(self.device["cmap"]))
        check_mapping(self.state)
        mapped = check_mapping.property_set["is_swap_mapped"]

        if mapped and self.layout is not None:
            return [self.action_terminate_index, *self.actions_opt_indices]  # type: ignore[unreachable]

        # No layout applied yet
        if self.state._layout is not None:
            return self.actions_routing_indices

        return self.actions_mapping_indices + self.actions_layout_indices + self.actions_opt_indices
