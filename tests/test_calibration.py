from __future__ import annotations

from mqt.predictor.Calibration import Calibration


def test_calibration() -> None:
    c = Calibration()
    assert c.ibm_montreal_calibration
    assert c.ibm_washington_calibration
    assert c.oqc_lucy_calibration
    assert c.rigetti_m2_calibration
    assert c.ionq_harmony_calibration
    assert c.ionq_aria1_calibration
    assert c.quantinuum_h2_calibration
