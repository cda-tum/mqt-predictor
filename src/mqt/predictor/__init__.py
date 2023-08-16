import logging
from mqt.predictor.Result import Result
from mqt.predictor.ml import qcompile

__all__ = [
    "Result",
    "qcompile",
]

logger = logging.getLogger("mqtpredictor")

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger_formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(logger_formatter)
logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)
