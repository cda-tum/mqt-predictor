from evaluator.src import utils
import pytest
from mqt.bench import get_one_benchmark

def test_get_machines():
    assert len(utils.get_machines()) == 10

def test_get_openqasm_gates():
    assert len(utils.get_openqasm_gates()) == 42

@pytest.mark.parametrize("backend", ["ibm_washington", "ibm_montreal", "ionq", "rigetti_m1", "oqc_lucy"])
def test_get_backend_information(backend:str):
    assert not utils.get_backend_information(backend) is None

def test_get_width_penalty():
    assert utils.get_width_penalty() >= 0