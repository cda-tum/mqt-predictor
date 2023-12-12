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
    """The Calibration class is used to store calibration data for different devices."""

    def __init__(self) -> None:
        try:
            self.ibm_montreal_calibration = FakeMontreal().properties()
            self.ibm_washington_calibration = FakeWashington().properties()
            self.oqc_lucy_calibration = parse_oqc_calibration_config()
            self.rigetti_m2_calibration = parse_rigetti_calibration_config()
            self.ionq_harmony_calibration = parse_simple_calibration_config("ionq_harmony")
            self.ionq_aria1_calibration = parse_simple_calibration_config("ionq_aria1")
            self.quantinuum_h2_calibration = parse_simple_calibration_config("quantinuum_h2")
            self.ibm_washington_cx_mean_error = get_mean_ibm_washington_cx_error()
            self.ibm_montreal_cx_mean_error = get_mean_ibm_montreal_cx_error()

        except Exception as e:
            raise RuntimeError("Error in Calibration initialization: " + str(e)) from e


def get_mean_ibm_washington_cx_error() -> float:
    """Returns the mean cx error for the IBM Washington device."""
    cmap: list[list[int]] = FakeWashington().configuration().coupling_map
    backend = FakeWashington().properties()
    nonfaulty_connections = [x for x in cmap if backend.gate_error("cx", x) < 1]

    res = [backend.gate_error("cx", elem) for elem in nonfaulty_connections]

    return cast(float, np.mean(res))


def get_mean_ibm_montreal_cx_error() -> float:
    """Returns the mean cx error for the IBM Washington device."""
    cmap: list[list[int]] = FakeMontreal().configuration().coupling_map
    backend = FakeMontreal().properties()
    nonfaulty_connections = [x for x in cmap if backend.gate_error("cx", x) < 1]

    res = [backend.gate_error("cx", elem) for elem in nonfaulty_connections]

    return cast(float, np.mean(res))


class DeviceCalibration(TypedDict):
    """The DeviceCalibration class is used to store calibration data for different devices."""

    backend: str
    avg_1Q: float
    avg_2Q: float


def parse_simple_calibration_config(device: str) -> DeviceCalibration:
    """Parses the calibration data for the given device.

    Args:
        device (str): The name of the device.

    Returns:
        DeviceCalibration: The calibration data for the given device.
    """

    calibration_filename = device + "_calibration.json"
    ref = resources.files("mqt.predictor") / "calibration_files" / calibration_filename
    with ref.open() as f:
        calibration = json.load(f)
    return {
        "backend": device,
        "avg_1Q": calibration["fidelity"]["1Q"].get("mean"),
        "avg_2Q": calibration["fidelity"]["2Q"].get("mean"),
    }


class OQCCalibration(DeviceCalibration):
    """The OQCCalibration class is used to store calibration data for the OQC device."""

    fid_1Q: dict[str, float]
    fid_1Q_readout: dict[str, float]
    fid_2Q: dict[str, float]


def parse_oqc_calibration_config() -> OQCCalibration:
    """Parses the calibration data for the OQC device.

    Returns:
        OQCCalibration: The calibration data for the OQC device.
    """
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
    """The RigettiCalibration class is used to store calibration data for the Rigetti Aspen M2 device."""

    fid_1Q: dict[str, float]
    fid_1Q_readout: dict[str, float]
    fid_2Q_CZ: dict[str, float]


def parse_rigetti_calibration_config() -> RigettiCalibration:
    """Parses the calibration data for the Rigetti Aspen M2 device.

    Returns:
        RigettiCalibration: The calibration data for the Rigetti Aspen M2 device.
    """
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
