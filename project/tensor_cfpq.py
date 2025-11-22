from typing import Set, Tuple

from networkx import DiGraph

from pyformlang.rsa import RecursiveAutomaton

from project.utils import graph_to_nfa, rsm_to_nfa

from project.adjacency_matrix_fa import AdjacencyMatrixFA, intersect_automata, ms_bfs


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
    nodes = set(graph.nodes)
    fa_graph = graph_to_nfa(graph, start_states=nodes, final_states=nodes)
    amfa_graph = AdjacencyMatrixFA(fa_graph)
    for var in rsm.labels:
        amfa_graph.add_symbol(var)
    adj_matrix_graph = amfa_graph.adjacency_matrices

    fa_rsm = rsm_to_nfa(rsm)
    amfa_rsm = AdjacencyMatrixFA(fa_rsm)
    m = amfa_rsm.count_states

    is_changed = True
    while is_changed:
        is_changed = False

        amfa = intersect_automata(amfa_graph, amfa_rsm)
        pairs = ms_bfs(amfa, amfa_rsm)

        for _, (s_id, f_id) in pairs:
            s, f = s_id // m, f_id // m
            var = amfa_rsm.get_state_by_id(s_id % m).value[0]
            if var and not adj_matrix_graph[var][s, f]:
                adj_matrix_graph[var][s, f] = True
                is_changed = True

    if start_nodes is None:
        start_nodes = nodes
    if final_nodes is None:
        final_nodes = nodes

    pairs = set()
    matrix = adj_matrix_graph[rsm.initial_label]

    for s_st_graph, s_id_graph in amfa_graph.states_ids.items():
        for f_st_graph, f_id_graph in amfa_graph.states_ids.items():
            if matrix[s_id_graph, f_id_graph]:
                if s_st_graph in start_nodes and f_st_graph in final_nodes:
                    pairs.add((s_st_graph.value, f_st_graph.value))

    return pairs
