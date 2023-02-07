from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal, TypedDict

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pytket.passes import BasePass as PytketBasePass  # type: ignore[attr-defined]
    from qiskit.transpiler.basepasses import BasePass as QiskitBasePass

import numpy as np
import requests
from mqt.predictor import SDK, rl
from mqt.predictor.devices import (
    Device,
    Provider,
    get_available_devices,
    get_available_providers,
)
from mqt.predictor.reward import calc_supermarq_features
from packaging import version
from pytket.architecture import Architecture  # type: ignore[attr-defined]
from pytket.circuit import OpType  # type: ignore[attr-defined]
from pytket.passes import (  # type: ignore[attr-defined]
    CliffordSimp,
    FullPeepholeOptimise,
    PeepholeOptimise2Q,
    RemoveRedundancies,
    RoutingPass,
)
from qiskit import QuantumCircuit
from qiskit.circuit.equivalence_library import StandardEquivalenceLibrary
from qiskit.circuit.library import XGate, ZGate
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
from sb3_contrib import MaskablePPO
from tqdm import tqdm

if TYPE_CHECKING or sys.version_info >= (3, 10, 0):
    from importlib import metadata, resources
else:
    import importlib_metadata as metadata
    import importlib_resources as resources

RewardFunction = Literal["fidelity", "critical_depth", "mix", "gate_ratio"]

logger = logging.getLogger("mqtpredictor")


def qcompile(qc: QuantumCircuit | Path, opt_objective: RewardFunction = "fidelity") -> tuple[QuantumCircuit, list[str]]:
    """Returns the compiled quantum circuit which is compiled following an objective function.
    Keyword arguments:
    qc -- to be compiled quantum circuit or path to a qasm file
    opt_objective -- objective function used for the compilation
    Returns: compiled quantum circuit as Qiskit QuantumCircuit object and the compilation steps used.
    """

    predictor = rl.Predictor()
    return predictor.compile_as_predicted(qc, opt_objective=opt_objective)


class Action(TypedDict):
    name: str


class CompilationAction(Action):
    origin: SDK


class OptimizationAction(CompilationAction):
    transpile_pass: list[QiskitBasePass | PytketBasePass]


def get_optimization_actions() -> list[OptimizationAction]:
    return [
        {
            "name": "Optimize1qGatesDecomposition",
            "transpile_pass": [Optimize1qGatesDecomposition()],
            "origin": SDK.qiskit,
        },
        {
            "name": "CXCancellation",
            "transpile_pass": [CXCancellation()],
            "origin": SDK.qiskit,
        },
        {
            "name": "CommutativeCancellation",
            "transpile_pass": [CommutativeCancellation()],
            "origin": SDK.qiskit,
        },
        {
            "name": "CommutativeInverseCancellation",
            "transpile_pass": [CommutativeInverseCancellation()],
            "origin": SDK.qiskit,
        },
        {
            "name": "RemoveDiagonalGatesBeforeMeasure",
            "transpile_pass": [RemoveDiagonalGatesBeforeMeasure()],
            "origin": SDK.qiskit,
        },
        {
            "name": "InverseCancellation",
            "transpile_pass": [InverseCancellation([XGate(), ZGate()])],
            "origin": SDK.qiskit,
        },
        {
            "name": "OptimizeCliffords",
            "transpile_pass": [OptimizeCliffords()],
            "origin": SDK.qiskit,
        },
        {
            "name": "Opt2qBlocks",
            "transpile_pass": [Collect2qBlocks(), ConsolidateBlocks()],
            "origin": SDK.qiskit,
        },
        {
            "name": "PeepholeOptimise2Q",
            "transpile_pass": [PeepholeOptimise2Q()],
            "origin": SDK.tket,
        },
        {
            "name": "CliffordSimp",
            "transpile_pass": [CliffordSimp()],
            "origin": SDK.tket,
        },
        {
            "name": "FullPeepholeOptimiseCX",
            "transpile_pass": [FullPeepholeOptimise(target_2qb_gate=OpType.TK2)],
            "origin": SDK.tket,
        },
        {
            "name": "RemoveRedundancies",
            "transpile_pass": [RemoveRedundancies()],
            "origin": SDK.tket,
        },
    ]


class MappingAction(CompilationAction):
    transpile_pass: Callable[[list[tuple[int, int]]], list[QiskitBasePass | PytketBasePass]]


def get_layout_actions() -> list[MappingAction]:
    return [
        {
            "name": "TrivialLayout",
            "transpile_pass": lambda c: [
                TrivialLayout(coupling_map=CouplingMap(c)),
                FullAncillaAllocation(coupling_map=CouplingMap(c)),
                EnlargeWithAncilla(),
                ApplyLayout(),
            ],
            "origin": SDK.qiskit,
        },
        {
            "name": "DenseLayout",
            "transpile_pass": lambda c: [
                DenseLayout(coupling_map=CouplingMap(c)),
                FullAncillaAllocation(coupling_map=CouplingMap(c)),
                EnlargeWithAncilla(),
                ApplyLayout(),
            ],
            "origin": SDK.qiskit,
        },
        {
            "name": "SabreLayout",
            "transpile_pass": lambda c: [
                SabreLayout(coupling_map=CouplingMap(c)),
                FullAncillaAllocation(coupling_map=CouplingMap(c)),
                EnlargeWithAncilla(),
                ApplyLayout(),
            ],
            "origin": SDK.qiskit,
        },
    ]


def get_routing_actions() -> list[MappingAction]:
    return [
        {
            "name": "BasicSwap",
            "transpile_pass": lambda c: [BasicSwap(coupling_map=CouplingMap(c))],
            "origin": SDK.qiskit,
        },
        {
            "name": "RoutingPass",
            "transpile_pass": lambda c: [
                RoutingPass(Architecture(c)),
            ],
            "origin": SDK.tket,
        },
        {
            "name": "StochasticSwap",
            "transpile_pass": lambda c: [StochasticSwap(coupling_map=CouplingMap(c))],
            "origin": SDK.qiskit,
        },
        {
            "name": "SabreSwap",
            "transpile_pass": lambda c: [SabreSwap(coupling_map=CouplingMap(c))],
            "origin": SDK.qiskit,
        },
    ]


class PlatformAction(Action):
    provider: Provider


def get_platform_actions() -> list[PlatformAction]:
    return [{"name": provider.provider_name, "provider": provider} for provider in get_available_providers()]


class SynthesisAction(CompilationAction):
    transpile_pass: Callable[[list[str]], list[QiskitBasePass | PytketBasePass]]


def get_synthesis_actions() -> list[SynthesisAction]:
    return [
        {
            "name": "BasisTranslator",
            "transpile_pass": lambda g: [BasisTranslator(StandardEquivalenceLibrary, target_basis=g)],
            "origin": SDK.qiskit,
        },
    ]


def get_termination_action() -> Action:
    return {"name": "terminate"}


class DeviceAction(Action):
    device: Device


def get_device_actions() -> list[DeviceAction]:
    return [
        {
            "name": device.name,
            "device": device,
        }
        for device in get_available_devices(sanitize_device=True)
    ]


def get_state_sample() -> QuantumCircuit:
    file_list = list(get_path_training_circuits().glob("*.qasm"))

    path_zip = get_path_training_circuits() / "mqtbench_sample_circuits.zip"
    if len(file_list) == 0 and path_zip.exists():
        import zipfile

        with zipfile.ZipFile(str(path_zip), "r") as zip_ref:
            zip_ref.extractall(get_path_training_circuits())

        file_list = list(get_path_training_circuits().glob("*.qasm"))
        assert len(file_list) > 0

    random_index = np.random.randint(len(file_list))
    try:
        qc = QuantumCircuit.from_qasm_file(str(file_list[random_index]))
    except Exception:
        raise RuntimeError("Could not read QuantumCircuit from: " + str(file_list[random_index])) from None

    return qc


class FeatureDict(TypedDict):
    num_qubits: NDArray[np.int_]
    depth: NDArray[np.int_]
    program_communication: NDArray[np.float32]
    critical_depth: NDArray[np.float32]
    entanglement_ratio: NDArray[np.float32]
    parallelism: NDArray[np.float32]
    liveness: NDArray[np.float32]


def create_feature_dict(qc: QuantumCircuit) -> FeatureDict:
    (
        program_communication,
        critical_depth,
        entanglement_ratio,
        parallelism,
        liveness,
    ) = calc_supermarq_features(qc)
    return {
        "num_qubits": np.array([qc.num_qubits], dtype=int),
        "depth": np.array([qc.depth()], dtype=int),
        "program_communication": np.array([program_communication], dtype=np.float32),
        "critical_depth": np.array([critical_depth], dtype=np.float32),
        "entanglement_ratio": np.array([entanglement_ratio], dtype=np.float32),
        "parallelism": np.array([parallelism], dtype=np.float32),
        "liveness": np.array([liveness], dtype=np.float32),
    }


def get_path_training_data() -> Path:
    return Path(str(resources.files("mqt.predictor"))) / "rl" / "training_data"


def get_path_trained_model() -> Path:
    return get_path_training_data() / "trained_model"


def get_path_training_circuits() -> Path:
    return get_path_training_data() / "training_circuits"


def load_model(model_name: str) -> MaskablePPO:
    path = get_path_trained_model()

    if Path(path / f"{model_name}.zip").exists():
        return MaskablePPO.load(path / f"{model_name}.zip")

    logger.info("Model does not exist. Try to retrieve suitable Model from GitHub...")
    try:
        mqtpredictor_module_version = metadata.version("mqt.predictor")
    except ModuleNotFoundError:
        error_msg = (
            "Could not retrieve version of mqt.predictor. Please run 'pip install . or pip install mqt.predictor'."
        )
        raise RuntimeError(error_msg) from None

    version_found = False
    response = requests.get("https://api.github.com/repos/cda-tum/mqtpredictor/tags")
    available_versions = []
    for elem in response.json():
        available_versions.append(elem["name"])
    for possible_version in available_versions:
        if version.parse(mqtpredictor_module_version) >= version.parse(possible_version):
            url = "https://api.github.com/repos/cda-tum/mqtpredictor/releases/tags/" + possible_version
            response = requests.get(url)
            if not response:
                error_msg = "Suitable trained models cannot be downloaded since the GitHub API failed. One reasons could be that the limit of 60 API calls per hour and IP address is exceeded."
                raise RuntimeError(error_msg)

            response_json = response.json()
            if "assets" in response_json:
                assets = response_json["assets"]
            elif "asset" in response_json:
                assets = [response_json["asset"]]
            else:
                assets = []

            for asset in assets:
                if model_name in asset["name"]:
                    version_found = True
                    download_url = asset["browser_download_url"]
                    logger.info("Downloading model from: " + download_url)
                    handle_downloading_model(download_url, model_name)
                    break

        if version_found:
            break

    if not version_found:
        error_msg = "No suitable model found on GitHub. Please update your mqt.predictort package using 'pip install -U mqt.predictor'."
        raise RuntimeError(error_msg) from None

    return MaskablePPO.load(path / model_name)


def handle_downloading_model(download_url: str, model_name: str) -> None:
    logger.info("Start downloading model...")

    r = requests.get(download_url)
    total_length = int(r.headers.get("content-length"))  # type: ignore[arg-type]
    fname = str(get_path_trained_model() / (model_name + ".zip"))

    with Path(fname).open(mode="wb") as f, tqdm(
        desc=fname,
        total=total_length,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in r.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)
    logger.info(f"Download completed to {fname}. ")
