from mqt.predictor import utils

from pathlib import Path

import numpy as np
from pytket import architecture
from pytket.circuit import OpType
from pytket.passes import (
    CliffordSimp,
    DecomposeMultiQubitsCX,
    FullPeepholeOptimise,
    PeepholeOptimise2Q,
    RemoveRedundancies,
    RoutingPass,
)
from qiskit import QuantumCircuit
from qiskit.circuit.equivalence_library import StandardEquivalenceLibrary
from qiskit.circuit.library import XGate, ZGate
from qiskit.providers.fake_provider import FakeMontreal, FakeWashington
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import (
    ApplyLayout,
    BasicSwap,
    BasisTranslator,
    Collect2qBlocks,
    CommutativeCancellation,
    CommutativeInverseCancellation,
    ConsolidateBlocks,
    CXCancellation,
    DenseLayout,
    EnlargeWithAncilla,
    FullAncillaAllocation,
    InverseCancellation,
    Optimize1qGatesDecomposition,
    OptimizeCliffords,
    RemoveDiagonalGatesBeforeMeasure,
    SabreLayout,
    SabreSwap,
    StochasticSwap,
    TrivialLayout,
)


def execute_TKET_FullPeephole(tket_qc, conn_write):
    FullPeepholeOptimise().apply(tket_qc)
    conn_write.send("success")


def execute_TKET_pass(tket_qc, tket_pass, conn_write):
    tket_pass.apply(tket_qc)
    conn_write.send("success")


def get_actions_opt():
    return [
        {
            "name": "Optimize1qGatesDecomposition",
            "transpile_pass": [Optimize1qGatesDecomposition()],
            "origin": "qiskit",
        },
        {
            "name": "CXCancellation",
            "transpile_pass": [CXCancellation()],
            "origin": "qiskit",
        },
        {
            "name": "CommutativeCancellation",
            "transpile_pass": [CommutativeCancellation()],
            "origin": "qiskit",
        },
        {
            "name": "CommutativeInverseCancellation",
            "transpile_pass": [CommutativeInverseCancellation()],
            "origin": "qiskit",
        },
        {
            "name": "RemoveDiagonalGatesBeforeMeasure",
            "transpile_pass": [RemoveDiagonalGatesBeforeMeasure()],
            "origin": "qiskit",
        },
        {
            "name": "InverseCancellation",
            "transpile_pass": [InverseCancellation([XGate(), ZGate()])],
            "origin": "qiskit",
        },
        {
            "name": "OptimizeCliffords",
            "transpile_pass": [OptimizeCliffords()],
            "origin": "qiskit",
        },
        {
            "name": "Opt2qBlocks",
            "transpile_pass": [Collect2qBlocks(), ConsolidateBlocks()],
            "origin": "qiskit",
        },
        {
            "name": "PeepholeOptimise2Q",
            "transpile_pass": [PeepholeOptimise2Q()],
            "origin": "tket",
        },
        {
            "name": "CliffordSimp",
            "transpile_pass": [CliffordSimp()],
            "origin": "tket",
        },
        {
            "name": "FullPeepholeOptimiseCX",
            "transpile_pass": [FullPeepholeOptimise(target_2qb_gate=OpType.CX)],
            "origin": "tket",
        },
        {
            "name": "RemoveRedundancies",
            "transpile_pass": [RemoveRedundancies()],
            "origin": "tket",
        },
    ]


def get_actions_layout():
    return [
        {
            "name": "TrivialLayout",
            "transpile_pass": lambda c: [
                TrivialLayout(coupling_map=CouplingMap(c)),
                FullAncillaAllocation(coupling_map=CouplingMap(c)),
                EnlargeWithAncilla(),
                ApplyLayout(),
            ],
            "origin": "qiskit",
        },
        {
            "name": "DenseLayout",
            "transpile_pass": lambda c: [
                DenseLayout(coupling_map=CouplingMap(c)),
                FullAncillaAllocation(coupling_map=CouplingMap(c)),
                EnlargeWithAncilla(),
                ApplyLayout(),
            ],
            "origin": "qiskit",
        },
        {
            "name": "SabreLayout_IBM_washington",
            "transpile_pass": lambda c: [
                SabreLayout(coupling_map=CouplingMap(c)),
                FullAncillaAllocation(coupling_map=CouplingMap(c)),
                EnlargeWithAncilla(),
                ApplyLayout(),
            ],
            "origin": "qiskit",
        },
    ]


def get_actions_routing():
    return [
        {
            "name": "BasicSwap",
            "transpile_pass": lambda c: [BasicSwap(coupling_map=CouplingMap(c))],
            "origin": "qiskit",
        },
        {
            "name": "RoutingPass",
            "transpile_pass": lambda c: [
                RoutingPass(architecture.Architecture(c)),
                DecomposeMultiQubitsCX(),
            ],
            "origin": "tket",
        },
        {
            "name": "StochasticSwap",
            "transpile_pass": lambda c: [StochasticSwap(coupling_map=CouplingMap(c))],
            "origin": "qiskit",
        },
        {
            "name": "SabreSwap",
            "transpile_pass": lambda c: [SabreSwap(coupling_map=CouplingMap(c))],
            "origin": "qiskit",
        },
    ]


def get_actions_platform_selection():
    return [
        {
            "name": "IBM",
            "gates": utils.get_ibm_native_gates(),
            "devices": ["ibm_washington", "ibm_montreal"],
            "max_qubit_size": 127,
        },
        {
            "name": "Rigetti",
            "gates": utils.get_rigetti_native_gates(),
            "devices": ["rigetti_aspen_m2"],
            "max_qubit_size": 80,
        },
        {
            "name": "OQC",
            "gates": utils.get_oqc_native_gates(),
            "devices": ["oqc_lucy"],
            "max_qubit_size": 8,
        },
        {
            "name": "IonQ",
            "gates": utils.get_ionq_native_gates(),
            "devices": ["ionq11"],
            "max_qubit_size": 11,
        },
    ]


def get_actions_synthesis():
    return [
        {
            "name": "BasisTranslator",
            "transpile_pass": lambda g: [
                BasisTranslator(StandardEquivalenceLibrary, target_basis=g)
            ],
            "origin": "qiskit",
        },
    ]


def get_action_terminate():
    return {"name": "terminate"}


def get_actions_devices():
    return [
        {
            "name": "ibm_washington",
            "transpile_pass": [],
            "full_connectivity": False,
            "cmap": utils.get_cmap_from_devicename("ibm_washington"),
            "max_qubits": 127,
        },
        {
            "name": "ibm_montreal",
            "transpile_pass": [],
            "device": "ibm_montreal",
            "full_connectivity": False,
            "cmap": utils.get_cmap_from_devicename("ibm_montreal"),
            "max_qubits": 27,
        },
        {
            "name": "oqc_lucy",
            "transpile_pass": [],
            "device": "oqc_lucy",
            "full_connectivity": False,
            "cmap": utils.get_cmap_from_devicename("oqc_lucy"),
            "max_qubits": 8,
        },
        {
            "name": "rigetti_aspen_m2",
            "transpile_pass": [],
            "device": "rigetti_aspen_m2",
            "full_connectivity": False,
            "cmap": utils.get_cmap_from_devicename("rigetti_aspen_m2"),
            "max_qubits": 80,
        },
        {
            "name": "ionq11",
            "transpile_pass": [],
            "device": "ionq11",
            "full_connectivity": True,
            "cmap": utils.get_cmap_from_devicename("ionq"),
            "max_qubits": 11,
        },
    ]


def get_random_state_sample():
    file_list = list(Path("./sample_circuits").glob("*.qasm"))
    random_index = np.random.randint(len(file_list))
    try:
        qc = QuantumCircuit.from_qasm_file(str(file_list[random_index]))
    except Exception:
        print("ERROR: ", file_list[random_index])
        return False
    return qc