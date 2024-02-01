from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import requests
from packaging import version
from pytket.architecture import Architecture
from pytket.circuit import Circuit, Node, Qubit
from pytket.passes import (
    CliffordSimp,
    FullPeepholeOptimise,
    PeepholeOptimise2Q,
    RemoveRedundancies,
    RoutingPass,
)
from pytket.placement import place_with_map
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
    Depth,
    EnlargeWithAncilla,
    FixedPoint,
    FullAncillaAllocation,
    GatesInBasis,
    InverseCancellation,
    MinimumPoint,
    Optimize1qGatesDecomposition,
    OptimizeCliffords,
    RemoveDiagonalGatesBeforeMeasure,
    SabreLayout,
    Size,
    StochasticSwap,
    TrivialLayout,
    UnitarySynthesis,
)
from sb3_contrib import MaskablePPO
from tqdm import tqdm

from mqt.bench.qiskit_helper import get_native_gates
from mqt.bench.utils import calc_supermarq_features, get_cmap_from_devicename
from mqt.predictor import reward, rl

if TYPE_CHECKING:
    from numpy.typing import NDArray

if TYPE_CHECKING or sys.version_info >= (3, 10, 0):
    from importlib import metadata, resources
else:
    import importlib_metadata as metadata
    import importlib_resources as resources

from bqskit import compile as bqskit_compile
from bqskit.ir import gates
from qiskit import QuantumRegister
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.preset_passmanagers import common
from qiskit.transpiler.runningpassmanager import ConditionalController

logger = logging.getLogger("mqt-predictor")


def qcompile(
    qc: QuantumCircuit | str,
    figure_of_merit: reward.figure_of_merit | None = "expected_fidelity",
    device_name: str | None = "ibm_washington",
    predictor_singleton: rl.Predictor | None = None,
) -> tuple[QuantumCircuit, list[str]]:
    """Compiles a given quantum circuit to a device optimizing for the given figure of merit.

    Args:
        qc (QuantumCircuit | str): The quantum circuit to be compiled. If a string is given, it is assumed to be a path to a qasm file.
        figure_of_merit (reward.reward_functions, optional): The figure of merit to be used for compilation. Defaults to "expected_fidelity".
        device_name (str, optional): The name of the device to compile to. Defaults to "ibm_washington".
        predictor_singleton (rl.Predictor, optional): A predictor object that is used for compilation to reduce compilation time when compiling multiple quantum circuits. If None, a new predictor object is created. Defaults to None.

    Returns:
        tuple[QuantumCircuit, list[str]] | bool: Returns a tuple containing the compiled quantum circuit and the compilation information. If compilation fails, False is returned.
    """

    if predictor_singleton is None:
        if figure_of_merit is None:
            msg = "figure_of_merit must not be None if predictor_singleton is None."
            raise ValueError(msg)
        if device_name is None:
            msg = "device_name must not be None if predictor_singleton is None."
            raise ValueError(msg)
        predictor = rl.Predictor(figure_of_merit=figure_of_merit, device_name=device_name)
    else:
        predictor = predictor_singleton

    return predictor.compile_as_predicted(qc)


def get_actions_opt() -> list[dict[str, Any]]:
    """Returns a list of dictionaries containing information about the optimization passes that are available."""
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
            "transpile_pass": [FullPeepholeOptimise()],
            "origin": "tket",
        },
        {
            "name": "RemoveRedundancies",
            "transpile_pass": [RemoveRedundancies()],
            "origin": "tket",
        },
        {
            "name": "QiskitO3",
            "transpile_pass": lambda native_gate, coupling_map: [
                Collect2qBlocks(),
                ConsolidateBlocks(basis_gates=native_gate),
                UnitarySynthesis(basis_gates=native_gate, coupling_map=coupling_map),
                Optimize1qGatesDecomposition(basis=native_gate),
                CommutativeCancellation(basis_gates=native_gate),
                GatesInBasis(native_gate),
                ConditionalController(
                    [
                        pass_
                        for x in common.generate_translation_passmanager(
                            target=None, basis_gates=native_gate, coupling_map=coupling_map
                        ).passes()
                        for pass_ in x["passes"]
                    ],
                    condition=lambda property_set: not property_set["all_gates_in_basis"],
                ),
                Depth(recurse=True),
                FixedPoint("depth"),
                Size(recurse=True),
                FixedPoint("size"),
                MinimumPoint(["depth", "size"], "optimization_loop"),
            ],
            "origin": "qiskit",
            "do_while": lambda property_set: (not property_set["optimization_loop_minimum_point"]),
        },
    ]


def get_actions_opt_before_layout() -> list[dict[str, Any]]:
    return [
        {
            "name": "BQSKitO2",
            "transpile_pass": lambda circuit: bqskit_compile(circuit, optimization_level=2),
            "origin": "bqskit",
        },
    ]


def get_actions_layout() -> list[dict[str, Any]]:
    """Returns a list of dictionaries containing information about the layout passes that are available."""
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
    ]


def get_actions_routing() -> list[dict[str, Any]]:
    """Returns a list of dictionaries containing information about the routing passes that are available."""
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
    ]


def get_actions_mapping() -> list[dict[str, Any]]:
    """Returns a list of dictionaries containing information about the mapping passes that are available."""
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
    """Returns a list of dictionaries containing information about the synthesis passes that are available."""
    return [
        {
            "name": "BasisTranslator",
            "transpile_pass": lambda g: [BasisTranslator(StandardEquivalenceLibrary, target_basis=g)],
            "origin": "qiskit",
        },
    ]


def get_action_terminate() -> dict[str, Any]:
    """Returns a dictionary containing information about the terminate pass that is available."""
    return {"name": "terminate"}


def get_devices() -> list[dict[str, Any]]:
    """Returns a list of dictionaries containing information about the devices that are available."""
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


def get_state_sample(max_qubits: int | None = None) -> tuple[QuantumCircuit, str]:
    """Returns a random quantum circuit from the training circuits folder.

    Args:
        max_qubits (int, None): The maximum number of qubits the returned quantum circuit may have. If no limit is set, it defaults to None.

    Returns:
        tuple[QuantumCircuit, str]: A tuple containing the random quantum circuit and the path to the file from which it was read.
    """
    file_list = list(get_path_training_circuits().glob("*.qasm"))

    path_zip = get_path_training_circuits() / "training_data_compilation.zip"
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
        if max_qubits and num_qubits > max_qubits:
            continue
        found_suitable_qc = True

    try:
        qc = QuantumCircuit.from_qasm_file(str(file_list[random_index]))
    except Exception:
        raise RuntimeError("Could not read QuantumCircuit from: " + str(file_list[random_index])) from None

    return qc, str(file_list[random_index])


def create_feature_dict(qc: QuantumCircuit) -> dict[str, int | NDArray[np.float_]]:
    """Creates a feature dictionary for a given quantum circuit.

    Args:
        qc (QuantumCircuit): The quantum circuit for which the feature dictionary is created.

    Returns:
        dict[str, Any]: The feature dictionary for the given quantum circuit.
    """

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
    """Returns the path to the training data folder used for RL training."""
    return Path(str(resources.files("mqt.predictor"))) / "rl" / "training_data"


def get_path_trained_model() -> Path:
    """Returns the path to the trained model folder used for RL training."""
    return get_path_training_data() / "trained_model"


def get_path_training_circuits() -> Path:
    """Returns the path to the training circuits folder used for RL training."""
    return get_path_training_data() / "training_circuits"


def load_model(model_name: str) -> MaskablePPO:
    """Loads a trained model from the trained model folder.

    Args:
        model_name (str): The name of the model to be loaded.

    Returns:
        MaskablePPO: The loaded model.
    """
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

    headers = None
    if "GITHUB_TOKEN" in os.environ:
        headers = {"Authorization": f"token {os.environ['GITHUB_TOKEN']}"}

    version_found = False
    response = requests.get("https://api.github.com/repos/cda-tum/mqt-predictor/tags", headers=headers)

    if not response:
        error_msg = "Querying the GitHub API failed. One reasons could be that the limit of 60 API calls per hour and IP address is exceeded."
        raise RuntimeError(error_msg)

    available_versions = [elem["name"] for elem in response.json()]

    for possible_version in available_versions:
        if version.parse(mqtpredictor_module_version) >= version.parse(possible_version):
            url = "https://api.github.com/repos/cda-tum/mqt-predictor/releases/tags/" + possible_version
            response = requests.get(url, headers=headers)
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
        error_msg = "No suitable model found on GitHub. Please update your mqt.predictor package using 'pip install -U mqt.predictor'."
        raise RuntimeError(error_msg) from None

    return MaskablePPO.load(path / model_name)


def handle_downloading_model(download_url: str, model_name: str) -> None:
    """Downloads a trained model from the given URL and saves it to the trained model folder.

    Args:
        download_url (str): The URL from which the model is downloaded.
        model_name (str): The name of the model to be downloaded.
    """
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
        Pre-processing step to route a circuit with TKET after a Qiskit Layout pass has been applied.
        The reason why we can apply the trivial layout here is that the circuit already got assigned a layout by qiskit.
        Implicitly, Qiskit is reordering its qubits in a sequential manner, i.e., the qubit with the lowest *physical* qubit
        first.

        Assuming, the layouted circuit is given by

                       ┌───┐           ░       ┌─┐
              q_2 -> 0 ┤ H ├──■────────░───────┤M├
                       └───┘┌─┴─┐      ░    ┌─┐└╥┘
              q_1 -> 1 ─────┤ X ├──■───░────┤M├─╫─
                            └───┘┌─┴─┐ ░ ┌─┐└╥┘ ║
              q_0 -> 2 ──────────┤ X ├─░─┤M├─╫──╫─
                                 └───┘ ░ └╥┘ ║  ║
        ancilla_0 -> 3 ───────────────────╫──╫──╫─
                                          ║  ║  ║
        ancilla_1 -> 4 ───────────────────╫──╫──╫─
                                          ║  ║  ║
               meas: 3/═══════════════════╩══╩══╩═
                                          0  1  2

        Applying the trivial layout, we get the same qubit order as in the original circuit and can be respectively
        routed. This results int:
                ┌───┐           ░       ┌─┐
           q_0: ┤ H ├──■────────░───────┤M├
                └───┘┌─┴─┐      ░    ┌─┐└╥┘
           q_1: ─────┤ X ├──■───░────┤M├─╫─
                     └───┘┌─┴─┐ ░ ┌─┐└╥┘ ║
           q_2: ──────────┤ X ├─░─┤M├─╫──╫─
                          └───┘ ░ └╥┘ ║  ║
           q_3: ───────────────────╫──╫──╫─
                                   ║  ║  ║
           q_4: ───────────────────╫──╫──╫─
                                   ║  ║  ║
        meas: 3/═══════════════════╩══╩══╩═
                                   0  1  2


        If we would not apply the trivial layout, no layout would be considered resulting, e.g., in the followiong circuit:
                 ┌───┐         ░    ┌─┐
       q_0: ─────┤ X ├─────■───░────┤M├───
            ┌───┐└─┬─┘   ┌─┴─┐ ░ ┌─┐└╥┘
       q_1: ┤ H ├──■───X─┤ X ├─░─┤M├─╫────
            └───┘      │ └───┘ ░ └╥┘ ║ ┌─┐
       q_2: ───────────X───────░──╫──╫─┤M├
                               ░  ║  ║ └╥┘
       q_3: ──────────────────────╫──╫──╫─
                                  ║  ║  ║
       q_4: ──────────────────────╫──╫──╫─
                                  ║  ║  ║
    meas: 3/══════════════════════╩══╩══╩═
                                  0  1  2

    """

    def apply(self, circuit: Circuit) -> None:
        """Applies the pre-processing step to route a circuit with tket after a Qiskit Layout pass has been applied."""
        mapping = {Qubit(i): Node(i) for i in range(circuit.n_qubits)}
        place_with_map(circuit=circuit, qmap=mapping)


def get_device(device_name: str) -> dict[str, Any]:
    """Returns the device with the given name.

    Args:
        device_name (str): The name of the device to be returned.

    Returns:
        dict[str, Any]: The device with the given name.
    """
    devices = get_devices()
    for device in devices:
        if device["name"] == device_name:
            return device

    msg = "No suitable device found."
    raise RuntimeError(msg)


def get_device_index_of_device(device_name: str) -> int:
    """Returns the index of the device with the given name.

    Args:
        device_name (str): The name of the device to be returned.

    Returns:
        int: The index of the device with the given name.
    """
    devices = get_devices()
    for i, device in enumerate(devices):
        if device["name"] == device_name:
            return i

    msg = "No suitable device found."
    raise RuntimeError(msg)


def get_BQSKit_native_gates(provider: str) -> list[gates.Gate] | None:
    """Returns the native gates of the given provider.

    Args:
        provider (str): The name of the provider.

    Returns:
        list[gates.Gate]: The native gates of the given provider.
    """

    native_gatesets = {
        "ibm": [gates.RZGate(), gates.SXGate(), gates.XGate(), gates.CNOTGate()],
        "rigetti": [gates.RXGate(), gates.RZGate(), gates.CZGate()],
        "ionq": [gates.RXXGate(), gates.RZGate(), gates.RYGate(), gates.RXGate()],
        "quantinuum": [gates.RZZGate(), gates.RZGate(), gates.RYGate(), gates.RXGate()],
    }

    if provider not in native_gatesets:
        logger.warning("No native gateset for provider " + provider + " found. No native gateset is used.")
        return None

    return native_gatesets[provider]


def final_layout_pytket_to_qiskit(pytket_circuit: Circuit) -> Layout:
    pytket_layout = pytket_circuit.qubit_readout
    size_circuit = pytket_circuit.n_qubits
    qiskit_layout = Layout()
    qiskit_qreg = QuantumRegister(size_circuit, "q")

    pytket_layout = dict(sorted(pytket_layout.items(), key=lambda item: item[1]))

    for node, qubit_index in pytket_layout.items():
        new_index = node.index[0]
        qiskit_layout[new_index] = qiskit_qreg[qubit_index]

    for i in range(size_circuit):
        if i not in set(pytket_layout.values()):
            qiskit_layout[i] = qiskit_qreg[i]

    return qiskit_layout
