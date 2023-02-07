from enum import Enum

import logging
from logging import StreamHandler

logger = logging.getLogger("mqtpredictor")

console_handler = StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger_formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(logger_formatter)
logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)


class SDK(Enum):
    qiskit = 0
    tket = 1


__all__ = ["SDK"]
