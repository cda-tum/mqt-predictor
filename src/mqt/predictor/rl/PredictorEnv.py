from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from pathlib import Path

    from mqt.predictor.rl import FeatureDict, RewardFunction
    from qiskit.transpiler.basepasses import BasePass

import numpy as np
from gym import Env
from gym.spaces import Box, Dict, Discrete
from mqt.predictor import SDK, reward
from mqt.predictor.devices import Device, Provider, get_available_providers
from mqt.predictor.rl.helper import (
    Action,
    CompilationAction,
    DeviceAction,
    MappingAction,
    OptimizationAction,
    PlatformAction,
    SynthesisAction,
    create_feature_dict,
    get_device_actions,
    get_layout_actions,
    get_optimization_actions,
    get_platform_actions,
    get_routing_actions,
    get_state_sample,
    get_synthesis_actions,
    get_termination_action,
)
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import CheckMap, GatesInBasis

logger = logging.getLogger("mqtpredictor")


class PredictorEnv(Env):  # type: ignore[misc]
    def __init__(self, reward_function: RewardFunction = "fidelity"):
        logger.info("Init env: " + reward_function)

        self.action_dict: dict[int, Action] = {}
        self.platform_action_indices: list[int] = []
        self.synthesis_action_indices: list[int] = []
        self.device_action_indices: list[int] = []
        self.layout_action_indices: list[int] = []
        self.routing_action_indices: list[int] = []
        self.optimization_action_indices: list[int] = []

        index = 0
        for platform_action in get_platform_actions():
            self.action_dict[index] = platform_action
            self.platform_action_indices.append(index)
            index += 1

        for device_action in get_device_actions():
            self.action_dict[index] = device_action
            self.device_action_indices.append(index)
            index += 1

        for synthesis_action in get_synthesis_actions():
            self.action_dict[index] = synthesis_action
            self.synthesis_action_indices.append(index)
            index += 1

        for layout_action in get_layout_actions():
            self.action_dict[index] = layout_action
            self.layout_action_indices.append(index)
            index += 1

        for routing_action in get_routing_actions():
            self.action_dict[index] = routing_action
            self.routing_action_indices.append(index)
            index += 1

        for optimization_action in get_optimization_actions():
            self.action_dict[index] = optimization_action
            self.optimization_action_indices.append(index)
            index += 1

        self.action_dict[index] = get_termination_action()
        self.action_terminate_index = index

        self.reward_function = reward_function
        self.action_space = Discrete(len(self.action_dict.keys()))
        self.num_steps = 0

        spaces = {
            "num_qubits": Discrete(max([provider.get_max_qubits() for provider in get_available_providers()])),
            "depth": Discrete(100000),
            "program_communication": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "critical_depth": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "entanglement_ratio": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "parallelism": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "liveness": Box(low=0, high=1, shape=(1,), dtype=np.float32),
        }
        self.observation_space = Dict(spaces)

        self.state: QuantumCircuit | None = None
        self.valid_actions: list[int] = []

        self.provider: Provider | None = None
        self.device: Device | None = None

    def step(self, action: int) -> tuple[FeatureDict, float, bool, dict[str, Any]]:
        altered_qc = self.apply_action(action)
        if not altered_qc:
            return create_feature_dict(self.state), 0, True, {}

        self.state = altered_qc
        self.num_steps += 1

        self.valid_actions = self.determine_valid_actions_for_state()
        if len(self.valid_actions) == 0:
            return create_feature_dict(self.state), 0, True, {}

        if action == self.action_terminate_index:
            reward_val = self.calculate_reward()
            done = True
        else:
            reward_val = 0
            done = False

        self.state = self.state.decompose(gates_to_decompose="unitary")
        return create_feature_dict(self.state), reward_val, done, {}

    def calculate_reward(self) -> float:
        assert self.device is not None
        if self.reward_function == "fidelity":
            return reward.expected_fidelity(self.state, self.device)
        if self.reward_function == "critical_depth":
            return reward.crit_depth(self.state)
        if self.reward_function == "mix":
            return reward.mix(self.state, self.device)
        if self.reward_function == "gate_ratio":
            return reward.gate_ratio(self.state)
        error_msg = f"Reward function {self.reward_function} not supported."  # type: ignore[unreachable]
        raise ValueError(error_msg)

    def render(self, mode: str = "text") -> Any:
        assert self.state is not None
        print(self.state.draw(output=mode))

    def reset(self, qc: Path | str | QuantumCircuit | None = None) -> FeatureDict:
        if isinstance(qc, QuantumCircuit):
            self.state = qc
        elif qc:
            self.state = QuantumCircuit.from_qasm_file(str(qc))
        else:
            self.state = get_state_sample()

        self.action_space = Discrete(len(self.action_dict.keys()))
        self.num_steps = 0
        self.device = None
        self.provider = None

        self.valid_actions = self.get_valid_platform_actions_for_state()

        return create_feature_dict(self.state)

    def action_masks(self) -> list[bool]:
        return [action in self.valid_actions for action in self.action_dict]

    def apply_action(self, action_index: int) -> QuantumCircuit:
        if action_index not in self.action_dict:
            msg = f"Action {action_index} does not exist."
            raise ValueError(msg)

        if action_index == self.action_terminate_index:
            return self.state

        if action_index in self.platform_action_indices:
            platform_action = cast(PlatformAction, self.action_dict[action_index])
            self.provider = platform_action["provider"]
            return self.state

        if action_index in self.device_action_indices:
            device_action = cast(DeviceAction, self.action_dict[action_index])
            self.device = device_action["device"]
            return self.state

        assert self.state is not None

        compilation_action = cast(CompilationAction, self.action_dict[action_index])
        origin = compilation_action["origin"]
        action_name = compilation_action["name"]
        if action_index in self.layout_action_indices + self.routing_action_indices:
            assert self.device is not None
            mapping_action = cast(MappingAction, self.action_dict[action_index])
            transpile_pass = mapping_action["transpile_pass"](self.device.coupling_map)
        elif action_index in self.synthesis_action_indices:
            assert self.device is not None
            synthesis_action = cast(SynthesisAction, self.action_dict[action_index])
            transpile_pass = synthesis_action["transpile_pass"](self.device.basis_gates)
        else:
            optimization_action = cast(OptimizationAction, self.action_dict[action_index])
            transpile_pass = optimization_action["transpile_pass"]

        if origin == SDK.qiskit:
            pm = PassManager(cast(list[BasePass], transpile_pass))
            try:
                return pm.run(self.state)
            except Exception as e:
                raise RuntimeError(
                    "Error in executing Qiskit transpile pass: " + action_name + ", " + self.state.name + ", " + str(e)
                ) from None

        if origin == SDK.tket:
            try:
                tket_qc = qiskit_to_tk(self.state)
                for elem in transpile_pass:
                    elem.apply(tket_qc)
                return tk_to_qiskit(tket_qc)
            except Exception as e:
                raise RuntimeError(
                    "Error in executing TKET transpile pass: " + action_name + ", " + self.state.name + ", " + str(e),
                ) from None

        msg = f"Origin {origin} not supported."
        raise ValueError(msg)

    def determine_valid_actions_for_state(self) -> list[int]:
        if self.provider is None:
            return self.get_valid_platform_actions_for_state() + self.optimization_action_indices

        if self.device is None:
            return self.get_valid_device_actions_for_state()

        check_nat_gates = GatesInBasis(basis_gates=self.device.basis_gates)
        check_nat_gates(self.state)
        only_nat_gates = check_nat_gates.property_set["all_gates_in_basis"]

        if not only_nat_gates:
            return self.synthesis_action_indices

        check_mapping = CheckMap(coupling_map=CouplingMap(self.device.coupling_map))
        check_mapping(self.state)
        mapped = check_mapping.property_set["is_swap_mapped"]

        if mapped:
            return [self.action_terminate_index, *self.optimization_action_indices]

        # No layout applied yet
        assert self.state is not None
        if self.state._layout is None:
            return self.layout_action_indices + self.optimization_action_indices

        return self.routing_action_indices + self.optimization_action_indices

    def get_valid_device_actions_for_state(self) -> list[int]:
        assert self.provider is not None
        assert self.state is not None
        return [
            action_index
            for action_index in self.device_action_indices
            if self.action_dict[action_index]["name"] in self.provider.get_available_device_names()
            and self.state.num_qubits <= cast(DeviceAction, self.action_dict[action_index])["device"].num_qubits
        ]

    def get_valid_platform_actions_for_state(self) -> list[int]:
        assert self.state is not None
        return [
            action_index
            for action_index in self.platform_action_indices
            if self.state.num_qubits
            <= cast(PlatformAction, self.action_dict[action_index])["provider"].get_max_qubits()
        ]
