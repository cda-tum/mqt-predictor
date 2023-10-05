from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qiskit import QuantumCircuit


class State:
    def __init__(self, qc: QuantumCircuit|None = None, path: str|None=None) -> None:
        self.qc = qc
        self.path = path
