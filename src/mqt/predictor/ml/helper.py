from __future__ import annotations

import sys

if sys.version_info < (3, 10, 0):
    import importlib_resources as resources
else:
    from importlib import resources  # type: ignore[no-redef]

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

from joblib import dump
from mqt.predictor import ml
from mqt.predictor.devices import Device, get_available_providers
from mqt.predictor.reward import calc_supermarq_features
from qiskit import QuantumCircuit

if TYPE_CHECKING:
    from sklearn.ensemble import RandomForestClassifier


def qcompile(qc: QuantumCircuit | Path) -> tuple[QuantumCircuit, int]:
    """Returns the compiled quantum circuit which is compiled with the predicted combination of compilation options.

    Keyword arguments:
    qc -- to be compiled quantum circuit or path to a qasm file

    Returns: compiled quantum circuit as Qiskit QuantumCircuit object
    """

    predictor = ml.Predictor()
    prediction = predictor.predict(qc)
    return predictor.compile_as_predicted(qc, prediction)


def get_path_training_data() -> Path:
    return Path(str(resources.files("mqt.predictor"))) / "ml" / "training_data"


def get_path_trained_model() -> Path:
    return get_path_training_data() / "trained_model"


def get_path_training_circuits() -> Path:
    return get_path_training_data() / "training_circuits"


def get_path_training_circuits_compiled() -> Path:
    return get_path_training_data() / "training_circuits_compiled"


def get_width_penalty() -> int:
    """Returns the penalty value if a quantum computer has not enough qubits."""
    return -10000


class QiskitOptions(TypedDict):
    optimization_level: int


class TketOptions(TypedDict):
    line_placement: bool


class CompilerOptions(TypedDict, total=False):
    qiskit: QiskitOptions
    tket: TketOptions


class CompilationPipeline(TypedDict):
    devices: dict[str, list[Device]]
    compiler: list[CompilerOptions]


def get_compilation_pipeline() -> CompilationPipeline:
    return {
        "devices": {provider.provider_name: provider.get_available_devices() for provider in get_available_providers()},
        "compiler": [
            {"qiskit": {"optimization_level": 0}},
            {"qiskit": {"optimization_level": 1}},
            {"qiskit": {"optimization_level": 2}},
            {"qiskit": {"optimization_level": 3}},
            {"tket": {"line_placement": False}},
            {"tket": {"line_placement": True}},
        ],
    }


class CompilationPath(TypedDict):
    provider_name: str
    device: Device
    compiler: str
    compiler_options: CompilerOptions


def get_index_to_compilation_path_dict() -> dict[int, CompilationPath]:
    compilation_pipeline = get_compilation_pipeline()
    index = 0
    index_to_compilation_path_dict = {}
    for provider_name, devices in compilation_pipeline["devices"].items():
        for device in devices:
            for configuration in compilation_pipeline["compiler"]:
                for compiler, _settings in configuration.items():
                    index_to_compilation_path_dict[index] = CompilationPath(
                        provider_name=provider_name,
                        device=device,
                        compiler=compiler,
                        compiler_options=configuration,
                    )
                    index += 1
    return index_to_compilation_path_dict


# according to https://github.com/Qiskit/qiskit-terra/blob/main/qiskit/qasm/libs/qelib1.inc
class OpenQASMGateCount(TypedDict):
    u3: float
    u2: float
    u1: float
    cx: float
    id: float  # noqa: A003
    u0: float
    u: float
    p: float
    x: float
    y: float
    z: float
    h: float
    s: float
    sdg: float
    t: float
    tdg: float
    rx: float
    ry: float
    rz: float
    sx: float
    sxdg: float
    cz: float
    cy: float
    swap: float
    ch: float
    ccx: float
    cswap: float
    crx: float
    cry: float
    crz: float
    cu1: float
    cp: float
    cu3: float
    csx: float
    cu: float
    rxx: float
    rzz: float
    rccx: float
    rc3x: float
    c3x: float
    c3sqrtx: float
    c4x: float


class CircuitFeatures(TypedDict):
    num_qubits: float
    depth: float
    program_communication: float
    critical_depth: float
    entanglement_ratio: float
    parallelism: float
    liveness: float


class FeatureDict(TypedDict):
    gate_count: OpenQASMGateCount
    circuit_features: CircuitFeatures


PATH_LENGTH = 260


def create_feature_dict(path: Path) -> FeatureDict:
    if Path(path).exists():
        qc = QuantumCircuit.from_qasm_file(str(path))
    else:
        error_msg = "Invalid input for 'qc' parameter."
        raise ValueError(error_msg) from None

    (
        program_communication,
        critical_depth,
        entanglement_ratio,
        parallelism,
        liveness,
    ) = calc_supermarq_features(qc)
    return {
        "gate_count": cast(
            OpenQASMGateCount,
            {
                gate.name: float(count)
                for gate, count in qc.count_ops().items()
                if gate.name in OpenQASMGateCount.__annotations__
            },
        ),
        "circuit_features": {
            "num_qubits": float(qc.num_qubits),
            "depth": float(qc.depth()),
            "program_communication": program_communication,
            "critical_depth": critical_depth,
            "entanglement_ratio": entanglement_ratio,
            "parallelism": parallelism,
            "liveness": liveness,
        },
    }


@dataclass
class TrainingSample:
    features: FeatureDict
    score: float

    def get_feature_vector(self) -> list[float]:
        """
        Returns the feature vector of the training sample.
        """
        return [cast(float, gate_count) for gate_count in self.features["gate_count"].values()] + [
            cast(float, circuit_feature) for circuit_feature in self.features["circuit_features"].values()
        ]


def save_classifier(clf: RandomForestClassifier) -> None:
    dump(clf, str(get_path_trained_model() / "trained_clf.joblib"))


def save_training_data(res: tuple[list[TrainingSample], list[str], list[list[float]]]) -> None:
    training_data, names_list, scores_list = res

    with resources.as_file(get_path_training_data() / "training_data_aggregated") as path:
        # pickle the data
        with Path(path / "training_data.pkl").open("wb") as f:
            pickle.dump(training_data, f)
        with Path(path / "names_list.pkl").open("wb") as f:
            pickle.dump(names_list, f)
        with Path(path / "scores_list.pkl").open("wb") as f:
            pickle.dump(scores_list, f)


def load_training_data() -> tuple[list[TrainingSample], list[str], list[list[float]]]:
    with resources.as_file(get_path_training_data() / "training_data_aggregated") as path:
        if (
            path.joinpath("training_data.pkl").is_file()
            and path.joinpath("names_list.pkl").is_file()
            and path.joinpath("scores_list.pkl").is_file()
        ):
            with Path(path / "training_data.pkl").open("rb") as f:
                training_data = pickle.load(f)
            with Path(path / "names_list.pkl").open("rb") as f:
                names_list = pickle.load(f)
            with Path(path / "scores_list.pkl").open("rb") as f:
                scores_list = pickle.load(f)
        else:
            error_msg = "Training data not found. Please run the training script first."
            raise FileNotFoundError(error_msg)

        return training_data, names_list, scores_list
