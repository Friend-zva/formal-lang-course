from typing import Set, Tuple

from networkx import DiGraph

from pyformlang.finite_automaton.finite_automaton import to_state, to_symbol

from pyformlang.rsa import RecursiveAutomaton

from project.utils import graph_to_nfa


def gll_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: DiGraph,
    start_nodes: Set[int] = None,
    final_nodes: Set[int] = None,
) -> Set[Tuple[int, int]]:
    pass
