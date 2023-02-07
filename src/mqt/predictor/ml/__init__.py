from mqt.predictor.ml import helper
from mqt.predictor.ml.helper import qcompile, TrainingSample, FeatureDict, QiskitOptions, TketOptions, CompilationPath
from mqt.predictor.ml.Predictor import Predictor


# TODO: This is a temporary solution. We should use the same backend names in MQT Bench.
BackendMapping: dict[str, str] = {
    "washington": "ibm_washington",
    "montreal": "ibm_montreal",
    "aspen-m2": "rigetti_aspen_m2",
    "harmony": "ionq11",
    "lucy": "oqc_lucy",
}


__all__ = [
    "helper",
    "qcompile",
    "Predictor",
    "TrainingSample",
    "FeatureDict",
    "QiskitOptions",
    "TketOptions",
    "CompilationPath",
    "BackendMapping",
]
