from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import requests
from mqt.bench.qiskit_helper import get_native_gates
from mqt.bench.utils import calc_supermarq_features, get_cmap_from_devicename
from mqt.predictor import rl
from packaging import version
from pytket.architecture import Architecture  # type: ignore[attr-defined]
from pytket.circuit import Circuit, Node, OpType, Qubit  # type: ignore[attr-defined]
from pytket.passes import (  # type: ignore[attr-defined]
    CliffordSimp,
    FullPeepholeOptimise,
    PeepholeOptimise2Q,
    RemoveRedundancies,
    RoutingPass,
)
from pytket.placement import place_with_map  # type: ignore[attr-defined]
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

reward_functions = Literal["fidelity", "critical_depth", "mix", "gate_ratio"]

logger = logging.getLogger("mqtpredictor")


def qcompile(
    qc: QuantumCircuit | str, opt_objective: reward_functions = "fidelity", device_name: str = "ibm"
) -> tuple[QuantumCircuit, list[str]]:
    """Returns the compiled quantum circuit which is compiled following an objective function.
    Keyword arguments:
    qc -- to be compiled quantum circuit or path to a qasm file
    opt_objective -- objective function used for the compilation
    Returns: compiled quantum circuit as Qiskit QuantumCircuit object
    """

    predictor = rl.Predictor()
    return predictor.compile_as_predicted(qc, opt_objective=opt_objective, device_name=device_name)


def get_actions_opt() -> list[dict[str, Any]]:
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
            "transpile_pass": [FullPeepholeOptimise(target_2qb_gate=OpType.TK2)],
            "origin": "tket",
        },
        {
            "name": "RemoveRedundancies",
            "transpile_pass": [RemoveRedundancies()],
            "origin": "tket",
        },
    ]


def get_actions_layout() -> list[dict[str, Any]]:
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
            "name": "SabreLayout",
            "transpile_pass": lambda c: [
                SabreLayout(coupling_map=CouplingMap(c), skip_routing=True),
                FullAncillaAllocation(coupling_map=CouplingMap(c)),
                EnlargeWithAncilla(),
                ApplyLayout(),
            ],
            "origin": "qiskit",
        },
    ]


def get_actions_routing() -> list[dict[str, Any]]:
    return [
        {
            "name": "BasicSwap",
            "transpile_pass": lambda c: [BasicSwap(coupling_map=CouplingMap(c))],
            "origin": "qiskit",
        },
        {
            "name": "RoutingPass",
            "transpile_pass": lambda c: [
                PreProcessTKETRoutingAfterQiskitLayout(),
                RoutingPass(Architecture(c)),
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


def get_actions_mapping() -> list[dict[str, Any]]:
    return [
        {
            "name": "SabreMapping",
            "transpile_pass": lambda c: [
                SabreLayout(coupling_map=CouplingMap(c), skip_routing=False),
            ],
            "origin": "qiskit",
        },
    ]


def get_actions_synthesis() -> list[dict[str, Any]]:
    return [
        {
            "name": "BasisTranslator",
            "transpile_pass": lambda g: [BasisTranslator(StandardEquivalenceLibrary, target_basis=g)],
            "origin": "qiskit",
        },
    ]


def get_action_terminate() -> dict[str, Any]:
    return {"name": "terminate"}


def get_platforms() -> list[dict[str, Any]]:
    return [
        {
            "name": "ibm",
            "gates": get_native_gates("ibm"),
            "devices": ["ibm_washington", "ibm_montreal"],
            "max_qubit_size": 127,
        },
        {
            "name": "rigetti",
            "gates": get_native_gates("rigetti"),
            "devices": ["rigetti_aspen_m2"],
            "max_qubit_size": 80,
        },
        {
            "name": "oqc",
            "gates": get_native_gates("oqc"),
            "devices": ["oqc_lucy"],
            "max_qubit_size": 8,
        },
        {
            "name": "ionq",
            "gates": get_native_gates("ionq"),
            "devices": ["ionq_harmony", "ionq_aria1"],
            "max_qubit_size": 25,
        },
        {
            "name": "quantinuum",
            "gates": get_native_gates("quantinuum"),
            "devices": ["quantinuum_h2"],
            "max_qubit_size": 32,
        },
    ]


def get_devices() -> list[dict[str, Any]]:
    return [
        {
            "name": "ibm_washington",
            "cmap": get_cmap_from_devicename("ibm_washington"),
            "native_gates": get_native_gates("ibm"),
            "max_qubits": 127,
        },
        {
            "name": "ibm_montreal",
            "cmap": get_cmap_from_devicename("ibm_montreal"),
            "native_gates": get_native_gates("ibm"),
            "max_qubits": 27,
        },
        {
            "name": "oqc_lucy",
            "cmap": get_cmap_from_devicename("oqc_lucy"),
            "native_gates": get_native_gates("oqc"),
            "max_qubits": 8,
        },
        {
            "name": "rigetti_aspen_m2",
            "cmap": get_cmap_from_devicename("rigetti_aspen_m2"),
            "native_gates": get_native_gates("rigetti"),
            "max_qubits": 80,
        },
        {
            "name": "ionq_harmony",
            "cmap": get_cmap_from_devicename("ionq_harmony"),
            "native_gates": get_native_gates("ionq"),
            "max_qubits": 11,
        },
        {
            "name": "ionq_aria1",
            "cmap": get_cmap_from_devicename("ionq_aria1"),
            "native_gates": get_native_gates("ionq"),
            "max_qubits": 25,
        },
        {
            "name": "quantinuum_h2",
            "cmap": get_cmap_from_devicename("quantinuum_h2"),
            "native_gates": get_native_gates("quantinuum"),
            "max_qubits": 32,
        },
    ]


def get_state_sample(max_qubits: int = -1) -> QuantumCircuit:
    file_list = list(get_path_training_circuits().glob("*.qasm"))

    path_zip = get_path_training_circuits() / "mqtbench_sample_circuits.zip"
    if len(file_list) == 0 and path_zip.exists():
        import zipfile

        with zipfile.ZipFile(str(path_zip), "r") as zip_ref:
            zip_ref.extractall(get_path_training_circuits())

        file_list = list(get_path_training_circuits().glob("*.qasm"))
        assert len(file_list) > 0

    found_suitable_qc = False
    while not found_suitable_qc:
        random_index = np.random.randint(len(file_list))
        num_qubits = int(str(file_list[random_index]).split("_")[-1].split(".")[0])
        if max_qubits > 0 and num_qubits > max_qubits:
            continue
        found_suitable_qc = True

    try:
        qc = QuantumCircuit.from_qasm_file(str(file_list[random_index]))
    except Exception:
        raise RuntimeError("Could not read QuantumCircuit from: " + str(file_list[random_index])) from None

    return qc, file_list[random_index]


def create_feature_dict(qc: QuantumCircuit) -> dict[str, Any]:
    feature_dict = {
        "num_qubits": qc.num_qubits,
        "depth": qc.depth(),
    }

    supermarq_features = calc_supermarq_features(qc)
    # for all dict values, put them in a list each
    feature_dict["program_communication"] = np.array([supermarq_features.program_communication], dtype=np.float32)
    feature_dict["critical_depth"] = np.array([supermarq_features.critical_depth], dtype=np.float32)
    feature_dict["entanglement_ratio"] = np.array([supermarq_features.entanglement_ratio], dtype=np.float32)
    feature_dict["parallelism"] = np.array([supermarq_features.parallelism], dtype=np.float32)
    feature_dict["liveness"] = np.array([supermarq_features.liveness], dtype=np.float32)

    return feature_dict


def get_path_training_data() -> Path:
    return Path(str(resources.files("mqt.predictor"))) / "rl" / "training_data"


def get_path_trained_model() -> Path:
    return get_path_training_data() / "trained_model"


def get_path_training_circuits() -> Path:
    return get_path_training_data() / "training_circuits"


def load_model(model_name: str) -> MaskablePPO:
    path = get_path_trained_model()

    if Path(path / (model_name + ".zip")).exists():
        return MaskablePPO.load(path / (model_name + ".zip"))
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


class PreProcessTKETRoutingAfterQiskitLayout:
    """
    Pre-processing step to route a circuit with tket after a Qiskit Layout pass has been applied.
    The reason why we can apply the trivial layout here is that the circuit is already mapped by qiskit to the
    device qubits and its qubits are sorted by their ascending physical qubit indices.
    The trivial layout indices that this layout of the physical qubits is the identity mapping.
    """

    def apply(self, circuit: Circuit) -> Circuit:
        mapping = {Qubit(i): Node(i) for i in range(circuit.n_qubits)}
        place_with_map(circuit=circuit, qmap=mapping)


def get_device(device_name: str) -> dict[str, Any]:
    devices = get_devices()
    for device in devices:
        if device["name"] == device_name:
            return device

    msg = "No suitable device found."
    raise RuntimeError(msg)
