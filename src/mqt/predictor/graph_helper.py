from __future__ import annotations

from typing import TYPE_CHECKING, Any

import networkx as nx  # type: ignore[import-untyped]
from qiskit.circuit import Clbit, Qubit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGInNode, DAGOutNode
from qiskit.transpiler.passes import RemoveBarriers
from torch_geometric.utils import from_networkx  # type: ignore[import-not-found]

if TYPE_CHECKING:
    import rustworkx as rx
    import torch_geometric  # type: ignore[import-not-found]
    from qiskit import QuantumCircuit


def rustworkx_to_networkx(graph: rx.PyDAG[Any, Any], ops_list_encoding: dict[str, int]) -> nx.MultiDiGraph | nx.DiGraph:
    """
    Convert a rustworkx DAG to a networkx graph.
    """
    # create a networkx graph
    nx_graph = nx.DiGraph()

    control_dict = {}  # to mark edges as control
    # add node operations = gates as nodes
    for node_idx, node in enumerate(graph.nodes()):
        if type(node) in [DAGInNode, DAGOutNode]:
            op = ops_list_encoding["id"]
        else:
            op = ops_list_encoding[node.op.name]

            controls = []  # mark controls
            if hasattr(node.op, "num_ctrl_qubits"):
                # last qubit is target
                controls += list(node.qargs[:-1])
            if node.op.condition:
                # classical c_if(...) operation
                controls += node.op.condition_bits
            if controls:
                control_dict[node_idx] = controls

        nx_graph.add_node(node_idx, gate=op)

    count = 0
    bit_dict: dict[Qubit | Clbit, int] = {}
    # add quantum/classical bits as edges
    for edge in graph.weighted_edge_list():
        source, target, bit = edge

        if not bit_dict.get(bit, False):
            bit_dict[bit] = count
            count += 1
        bit_nr = bit_dict[bit]

        is_classic = 1 if isinstance(bit, Clbit) else 0

        is_control = 1 if bit in control_dict.get(target, []) else 0

        nx_graph.add_edge(source, target, bit_nr=bit_nr, is_classic=is_classic, is_control=is_control)

    return nx_graph


def circuit_to_graph(qc: QuantumCircuit, ops_list_encoding: dict[str, int]) -> torch_geometric.data.Data:
    """
    Convert a quantum circuit to a torch_geometric graph.
    """
    ### Preprocessing ###
    circ = RemoveBarriers()(qc)

    # Convert to a rustworkx DAG
    dag_circuit = circuit_to_dag(circ, copy_operations=False)
    dag_graph = dag_circuit._multi_graph  # noqa: SLF001
    # Convert to networkx graph
    nx_graph = rustworkx_to_networkx(dag_graph, ops_list_encoding)

    #### Postprocessing ###
    # Remove root and leaf nodes (in and out nodes)
    nodes_to_remove = [node for node, degree in nx_graph.degree() if degree == 1]
    nx_graph.remove_nodes_from(nodes_to_remove)
    # Remove edges where is_classic == 1 (classical wires)
    edges_to_remove = [(u, v) for u, v, attr in nx_graph.edges(data=True) if attr.get("is_classic") == 1]
    nx_graph.remove_edges_from(edges_to_remove)

    # Convert to torch_geometric data
    return from_networkx(nx_graph, group_node_attrs=all, group_edge_attrs=all)
