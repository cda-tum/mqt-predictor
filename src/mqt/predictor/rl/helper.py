from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import requests
from mqt.bench.utils import calc_supermarq_features
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
from sb3_contrib import MaskablePPO
from tqdm import tqdm

if TYPE_CHECKING or sys.version_info >= (3, 10, 0):
    from importlib import metadata, resources
else:
    import importlib_metadata as metadata
    import importlib_resources as resources

reward_functions = Literal["fidelity", "critical_depth", "mix", "gate_ratio"]

logger = logging.getLogger("mqtpredictor")


def qcompile(qc: QuantumCircuit | str, opt_objective: reward_functions = "fidelity") -> QuantumCircuit:
    """Returns the compiled quantum circuit which is compiled following an objective function.
    Keyword arguments:
    qc -- to be compiled quantum circuit or path to a qasm file
    opt_objective -- objective function used for the compilation
    Returns: compiled quantum circuit as Qiskit QuantumCircuit object
    """

    predictor = rl.Predictor()
    return predictor.compile_as_predicted(qc, opt_objective=opt_objective)


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


def get_actions_platform_selection() -> list[dict[str, Any]]:
    return [
        {
            "name": "IBM",
            "gates": get_ibm_native_gates(),
            "devices": ["ibm_washington", "ibm_montreal"],
            "max_qubit_size": 127,
        },
        {
            "name": "Rigetti",
            "gates": get_rigetti_native_gates(),
            "devices": ["rigetti_aspen_m2"],
            "max_qubit_size": 80,
        },
        {
            "name": "OQC",
            "gates": get_oqc_native_gates(),
            "devices": ["oqc_lucy"],
            "max_qubit_size": 8,
        },
        {
            "name": "IonQ",
            "gates": get_ionq_native_gates(),
            "devices": ["ionq11"],
            "max_qubit_size": 11,
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


def get_actions_devices() -> list[dict[str, Any]]:
    return [
        {
            "name": "ibm_washington",
            "transpile_pass": [],
            "full_connectivity": False,
            "cmap": get_cmap_from_devicename("ibm_washington"),
            "max_qubits": 127,
        },
        {
            "name": "ibm_montreal",
            "transpile_pass": [],
            "device": "ibm_montreal",
            "full_connectivity": False,
            "cmap": get_cmap_from_devicename("ibm_montreal"),
            "max_qubits": 27,
        },
        {
            "name": "oqc_lucy",
            "transpile_pass": [],
            "device": "oqc_lucy",
            "full_connectivity": False,
            "cmap": get_cmap_from_devicename("oqc_lucy"),
            "max_qubits": 8,
        },
        {
            "name": "rigetti_aspen_m2",
            "transpile_pass": [],
            "device": "rigetti_aspen_m2",
            "full_connectivity": False,
            "cmap": get_cmap_from_devicename("rigetti_aspen_m2"),
            "max_qubits": 80,
        },
        {
            "name": "ionq11",
            "transpile_pass": [],
            "device": "ionq11",
            "full_connectivity": True,
            "cmap": get_cmap_from_devicename("ionq11"),
            "max_qubits": 11,
        },
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


def get_ibm_native_gates() -> list[str]:
    return ["rz", "sx", "x", "cx", "measure"]


def get_rigetti_native_gates() -> list[str]:
    return ["rx", "rz", "cz", "measure"]


def get_ionq_native_gates() -> list[str]:
    return ["rxx", "rz", "ry", "rx", "measure"]


def get_oqc_native_gates() -> list[str]:
    return ["rz", "sx", "x", "ecr", "measure"]


def get_rigetti_aspen_m2_map() -> list[list[int]]:
    """Returns a coupling map of Rigetti Aspen M2 chip."""
    c_map_rigetti = []
    for j in range(5):
        for i in range(7):
            c_map_rigetti.append([i + j * 8, i + 1 + j * 8])

            if i == 6:
                c_map_rigetti.append([0 + j * 8, 7 + j * 8])

        if j != 0:
            c_map_rigetti.append([j * 8 - 6, j * 8 + 5])
            c_map_rigetti.append([j * 8 - 7, j * 8 + 6])

    for j in range(5):
        m = 8 * j + 5 * 8
        for i in range(7):
            c_map_rigetti.append([i + m, i + 1 + m])

            if i == 6:
                c_map_rigetti.append([0 + m, 7 + m])

        if j != 0:
            c_map_rigetti.append([m - 6, m + 5])
            c_map_rigetti.append([m - 7, m + 6])

    for n in range(5):
        c_map_rigetti.append([n * 8 + 3, n * 8 + 5 * 8])
        c_map_rigetti.append([n * 8 + 4, n * 8 + 7 + 5 * 8])

    inverted = [[item[1], item[0]] for item in c_map_rigetti]

    return c_map_rigetti + inverted


def get_ionq11_c_map() -> list[list[int]]:
    ionq11_c_map = []
    for i in range(11):
        for j in range(11):
            if i != j:
                ionq11_c_map.append([i, j])
    return ionq11_c_map


def get_cmap_oqc_lucy() -> list[list[int]]:
    """Returns the coupling map of the OQC Lucy quantum computer."""
    # source: https://github.com/aws/amazon-braket-examples/blob/main/examples/braket_features/Verbatim_Compilation.ipynb

    # Connections are NOT bidirectional, this is not an accident
    return [[0, 1], [0, 7], [1, 2], [2, 3], [7, 6], [6, 5], [4, 3], [4, 5]]


def get_cmap_from_devicename(device: str) -> Any:
    if device == "ibm_washington":
        return FakeWashington().configuration().coupling_map
    if device == "ibm_montreal":
        return FakeMontreal().configuration().coupling_map
    if device == "rigetti_aspen_m2":
        return get_rigetti_aspen_m2_map()
    if device == "oqc_lucy":
        return get_cmap_oqc_lucy()
    if device == "ionq11":
        return get_ionq11_c_map()
    error_msg = "Unknown device name"
    raise ValueError(error_msg)


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
