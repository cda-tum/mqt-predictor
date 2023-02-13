from __future__ import annotations

import json
import sys

if sys.version_info < (3, 10, 0):
    import importlib_resources as resources
else:
    from importlib import resources  # type: ignore[no-redef]
from typing import TypedDict, cast

import numpy as np
from qiskit.providers.fake_provider import FakeMontreal, FakeWashington


class Calibration:
    def __init__(self) -> None:
        try:
            self.ibm_washington_cx_mean_error = get_mean_IBM_washington_cx_error()
            self.ibm_montreal_calibration = FakeMontreal().properties()
            self.ibm_washington_calibration = FakeWashington().properties()
            self.oqc_lucy_calibration = parse_oqc_calibration_config()
            self.rigetti_m2_calibration = parse_rigetti_calibration_config()
            self.ionq_calibration = parse_ionq_calibration_config()

        except Exception as e:
            raise RuntimeError("Error in Calibration initialization: " + str(e)) from e


def get_mean_IBM_washington_cx_error() -> float:
    cmap: list[list[int]] = FakeWashington().configuration().coupling_map
    backend = FakeWashington().properties()
    somelist = [x for x in cmap if backend.gate_error("cx", x) < 1]

    res: list[float] = []
    for elem in somelist:
        res.append(backend.gate_error("cx", elem))

    return cast(float, np.mean(res))


class DeviceCalibration(TypedDict):
    backend: str
    avg_1Q: float
    avg_2Q: float


def parse_ionq_calibration_config() -> DeviceCalibration:
    ref = resources.files("mqt.predictor") / "calibration_files" / "ionq_calibration.json"
    with ref.open() as f:
        ionq_calibration = json.load(f)
    return {
        "backend": "ionq",
        "avg_1Q": ionq_calibration["fidelity"]["1Q"].get("mean"),
        "avg_2Q": ionq_calibration["fidelity"]["2Q"].get("mean"),
    }


class OQCCalibration(DeviceCalibration):
    fid_1Q: dict[str, float]
    fid_1Q_readout: dict[str, float]
    fid_2Q: dict[str, float]


def parse_oqc_calibration_config() -> OQCCalibration:
    ref = resources.files("mqt.predictor") / "calibration_files" / "oqc_lucy_calibration.json"
    with ref.open() as f:
        oqc_lucy_calibration = json.load(f)
    fid_1Q = {}
    fid_1Q_readout = {}
    for elem in oqc_lucy_calibration["oneQubitProperties"]:
        fid_1Q[str(elem)] = oqc_lucy_calibration["oneQubitProperties"][elem]["oneQubitFidelity"][0].get("fidelity")
        fid_1Q_readout[str(elem)] = oqc_lucy_calibration["oneQubitProperties"][elem]["oneQubitFidelity"][1].get(
            "fidelity"
        )
    fid_2Q = {}
    for elem in oqc_lucy_calibration["twoQubitProperties"]:
        fid_2Q[str(elem)] = oqc_lucy_calibration["twoQubitProperties"][elem]["twoQubitGateFidelity"][0].get("fidelity")

    avg_1Q = np.average(list(fid_1Q.values()))
    avg_2Q = np.average(list(fid_2Q.values()))
    return {
        "backend": "oqc_lucy",
        "avg_1Q": avg_1Q,
        "fid_1Q": fid_1Q,
        "fid_1Q_readout": fid_1Q_readout,
        "avg_2Q": avg_2Q,
        "fid_2Q": fid_2Q,
    }


class RigettiCalibration(DeviceCalibration):
    fid_1Q: dict[str, float]
    fid_1Q_readout: dict[str, float]
    fid_2Q_CZ: dict[str, float]


def parse_rigetti_calibration_config() -> RigettiCalibration:
    ref = resources.files("mqt.predictor") / "calibration_files" / "rigetti_m2_calibration.json"
    with ref.open() as f:
        rigetti_m2_calibration = json.load(f)
    fid_1Q = {}
    fid_1Q_readout = {}
    missing_indices: list[int] = []
    for elem in rigetti_m2_calibration["specs"]["1Q"]:
        fid_1Q[str(elem)] = rigetti_m2_calibration["specs"]["1Q"][elem].get("f1QRB")
        fid_1Q_readout[str(elem)] = rigetti_m2_calibration["specs"]["1Q"][elem].get("fRO")

    fid_2Q_CZ = {}
    non_list = []
    for elem in rigetti_m2_calibration["specs"]["2Q"]:
        if rigetti_m2_calibration["specs"]["2Q"][elem].get("fCZ") is None:
            non_list.append(elem)
        else:
            fid_2Q_CZ[str(elem)] = rigetti_m2_calibration["specs"]["2Q"][elem].get("fCZ")

    cz_fid_avg = np.average(list(fid_2Q_CZ.values()))

    avg_1Q = np.average(list(fid_1Q.values()))
    for elem in missing_indices:
        fid_2Q_CZ[elem] = cz_fid_avg

    return {
        "backend": "rigetti_aspen_m2",
        "avg_1Q": avg_1Q,
        "fid_1Q": fid_1Q,
        "fid_1Q_readout": fid_1Q_readout,
        "avg_2Q": cz_fid_avg,
        "fid_2Q_CZ": fid_2Q_CZ,
    }
