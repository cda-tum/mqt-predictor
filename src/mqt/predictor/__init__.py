"""MQT Predictor.

This file is part of the MQT Predictor library released under the MIT license.
See README.md or go to https://github.com/cda-tum/mqt-predictor for more information.
"""

from __future__ import annotations

import logging

from mqt.predictor.ml import qcompile
from mqt.predictor.result import Result

__all__ = [
    "Result",
    "qcompile",
]

logger = logging.getLogger("mqt-predictor")

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger_formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(logger_formatter)
logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)
