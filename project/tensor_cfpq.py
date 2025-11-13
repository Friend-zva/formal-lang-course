from typing import Set, Tuple, Dict
from pyformlang.rsa import RecursiveAutomaton
from pyformlang.cfg import CFG
from networkx import DiGraph

from scipy.sparse import lil_array

from project.adjacency_matrix_fa import AdjacencyMatrixFA, intersect_automata


def cfg_to_rsm(cfg: CFG) -> RecursiveAutomaton:
    cfg_str = cfg.to_text()
    return RecursiveAutomaton.from_text(cfg_str)


def ebnf_to_rsm(ebnf: str) -> RecursiveAutomaton:
    return RecursiveAutomaton.from_text(ebnf)


def tensor_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: DiGraph,
    start_nodes: Set[int] = None,
    final_nodes: Set[int] = None,
) -> Set[Tuple[int, int]]:
    """Evaluate CFG path queries using the tensor algorithm

    Parameters
    ----------
    cfg : :class:`~pyformlang.rsa.RecursiveAutomaton`
        CFG defining the path constraint
    graph : :class:`~networkx.DiGraph`
        Graph where edges are labeled with symbols
    start_nodes : Set[int]
        Set of start nodes
    final_nodes : Set[int]
        Set of final nodes

    Returns
    -------
    pairs : Set[Tuple[int, int]]
        Set of node pairs (start, final) connected by paths matching CFG
    """

    pass
