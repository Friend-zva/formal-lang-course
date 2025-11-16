from typing import Set, Tuple
from pyformlang.rsa import RecursiveAutomaton
from pyformlang.cfg import CFG
from pyformlang.finite_automaton import (
    NondeterministicFiniteAutomaton,
    DeterministicFiniteAutomaton,
)
from pyformlang.finite_automaton.finite_automaton import to_state, to_symbol
from networkx import DiGraph

from project.adjacency_matrix_fa import AdjacencyMatrixFA, intersect_automata, ms_bfs
from project.utils import graph_to_nfa


def cfg_to_rsm(cfg: CFG) -> RecursiveAutomaton:
    """Transforms the CFG into RSM

    Parameters
    ----------
    cfg : :class:`~pyformlang.cfg.CFG`
        Context-Free Grammar

    Returns
    -------
    rsm : :class:`~pyformlang.rsa.RecursiveAutomaton`
        Recursive State Machine equivalent to the CFG
    """
    cfg_str = cfg.to_text()
    return RecursiveAutomaton.from_text(cfg_str)


def ebnf_to_rsm(ebnf: str) -> RecursiveAutomaton:
    """Transforms the EBNF into RSM

    Parameters
    ----------
    ebnf : str
        Extended Backus-Naur Form

    Returns
    -------
    rsm : :class:`~pyformlang.rsa.RecursiveAutomaton`
        Recursive State Machine equivalent to the EBNF
    """
    return RecursiveAutomaton.from_text(ebnf)


def rsm_to_nfa(rsm: RecursiveAutomaton) -> NondeterministicFiniteAutomaton:
    """Transforms the RSM into NFA

    Parameters
    ----------
    rsm : :class:`~pyformlang.rsa.RecursiveAutomaton`
        Recursive State Machine

    Returns
    -------
    nfa : :class:`~pyformlang.finite_automaton.NondeterministicFiniteAutomaton`
        Nondeterministic Finite Automaton equivalent to the RSM
    """
    nfa = NondeterministicFiniteAutomaton()

    for var, box in rsm.boxes.items():
        dfa: DeterministicFiniteAutomaton = box.dfa

        for state in dfa.start_states:
            state = to_state((var, state))
            nfa.add_start_state(state)

        for state in dfa.final_states:
            state = to_state((var, state))
            nfa.add_final_state(state)

        graph = dfa.to_networkx()
        for src, dst, sym in graph.edges(data="label"):
            if sym:
                src = to_state((var, src))
                dst = to_state((var, dst))
                sym = to_symbol(sym)
                nfa.add_transition(src, sym, dst)

    return nfa


def tensor_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: DiGraph,
    start_nodes: Set[int] = None,
    final_nodes: Set[int] = None,
) -> Set[Tuple[int, int]]:
    """Evaluate RSM path queries using the tensor algorithm

    Parameters
    ----------
    rsm : :class:`~pyformlang.rsa.RecursiveAutomaton`
        Recursive State Machine defining the path constraint
    graph : :class:`~networkx.DiGraph`
        Graph where edges are labeled with symbols
    start_nodes : Set[int]
        Set of start nodes
    final_nodes : Set[int]
        Set of final nodes

    Returns
    -------
    pairs : Set[Tuple[int, int]]
        Set of node pairs (start, final) connected by paths matching RSM
    """
    if start_nodes is None:
        start_nodes = set(graph.nodes)
    if final_nodes is None:
        final_nodes = set(graph.nodes)

    fa_graph = graph_to_nfa(graph, start_nodes, final_nodes)
    amfa_graph = AdjacencyMatrixFA(fa_graph)
    for var in rsm.labels:
        amfa_graph.add_symbol(var)
    adj_matrix_graph = amfa_graph.adjacency_matrices
    n = amfa_graph.count_states

    fa_rsm = rsm_to_nfa(rsm)
    amfa_rsm = AdjacencyMatrixFA(fa_rsm)
    m = amfa_rsm.count_states

    is_changed = True
    while is_changed:
        is_changed = False

        amfa = intersect_automata(amfa_graph, amfa_rsm)
        pairs = ms_bfs(amfa, amfa_rsm)

        for _, (s_id, f_id) in pairs:
            s_id = s_id // m
            f_id = f_id % n
            for symbol in amfa_graph.symbols:
                if not adj_matrix_graph[symbol][s_id, f_id]:
                    adj_matrix_graph[symbol][s_id, f_id] = True
                    is_changed = True

    pairs = set()
    matrix = adj_matrix_graph[rsm.initial_label]

    for s_st_graph, s_id_graph in amfa_graph.start_states_ids.items():
        for f_st_graph, f_id_graph in amfa_graph.final_states_ids.items():
            if matrix[s_id_graph, f_id_graph]:
                if s_st_graph in start_nodes and f_st_graph in final_nodes:
                    pairs.add((s_st_graph.value, f_st_graph.value))

    return pairs
