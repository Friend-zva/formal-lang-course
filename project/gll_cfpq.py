from collections import defaultdict
from typing import Set, Tuple
from pyformlang.rsa import RecursiveAutomaton
from pyformlang.cfg import CFG
from pyformlang.finite_automaton import (
    NondeterministicFiniteAutomaton,
    DeterministicFiniteAutomaton,
)
from pyformlang.finite_automaton.finite_automaton import to_state, to_symbol
from networkx import DiGraph
from scipy import sparse

from project.adjacency_matrix_fa import AdjacencyMatrixFA, intersect_automata, ms_bfs
from project.utils import graph_to_nfa


def gll_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: DiGraph,
    start_nodes: Set[int] = None,
    final_nodes: Set[int] = None,
) -> Set[Tuple[int, int]]:
    pass
