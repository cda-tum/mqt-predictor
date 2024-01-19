from __future__ import annotations

from mqt.predictor import Result


def test_result_none_input() -> None:
    res = Result("test", 1.0, None, None)
    assert res.compilation_time == 1.0
    assert res.fidelity == -1.0
    assert res.critical_depth == -1.0
