from pyformlang.finite_automaton import (
    NondeterministicFiniteAutomaton,
    DeterministicFiniteAutomaton,
)
from scipy.sparse import lil_array
from networkx import MultiDiGraph

from project.adjacency_matrix_fa import (
    AdjacencyMatrixFA,
    intersect_automata,
    tensor_based_rpq,
    ms_bfs_based_rpq,
)
from project.utils import regex_to_dfa


def test_adjacency_matrix_fa_empty():
    amfa = AdjacencyMatrixFA()
    assert amfa
    assert not amfa.accepts("a")
    assert amfa.is_empty()


def test_adjacency_matrix_fa_from_arg():
    amfa = AdjacencyMatrixFA(
        states_ids={0: 0, 1: 1, 2: 2, 3: 3},
        symbols={"a", "b", "c"},
        start_states={0},
        final_states={2, 3},
        adjacency_matrices=dict(
            [
                (
                    "a",
                    lil_array(
                        [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
                        dtype=bool,
                    ),
                ),
                (
                    "b",
                    lil_array(
                        [[0, 1, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
                        dtype=bool,
                    ),
                ),
                (
                    "c",
                    lil_array(
                        [[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]],
                        dtype=bool,
                    ),
                ),
            ]
        ),
    )

    assert not amfa.accepts("a")
    assert not amfa.is_empty()
    assert amfa.accepts("aaacbcaac")


def test_adjacency_matrix_fa_from_dfa():
    dfa = DeterministicFiniteAutomaton()
    dfa.add_transitions(
        [
            (0, "a", 1),
            (1, "b", 2),
            (2, "c", 3),
            (3, "b", 4),
            (2, "d", 4),
        ]
    )
    start_state = 0
    final_state = 4
    dfa.add_start_state(start_state)
    dfa.add_final_state(final_state)

    amfa = AdjacencyMatrixFA(dfa)
    assert not amfa.is_empty()
    assert amfa.accepts("abcb")


def test_adjacency_matrix_fa_from_nfa():
    nfa = NondeterministicFiniteAutomaton()
    nfa.add_transitions(
        [
            (0, "a", 1),
            (0, "a", 0),
            (1, "b", 2),
            (2, "c", 3),
            (3, "b", 4),
            (2, "d", 4),
        ]
    )
    start_state = 0
    final_state = 4
    nfa.add_start_state(start_state)
    nfa.add_final_state(final_state)

    amfa = AdjacencyMatrixFA(nfa)
    assert not amfa.is_empty()
    assert amfa.accepts("aabcb")


def test_intersect_automata():
    dfa1 = AdjacencyMatrixFA(regex_to_dfa("(a a)*"))
    dfa2 = AdjacencyMatrixFA(regex_to_dfa("a*b*"))
    amfa = intersect_automata(dfa1, dfa2)

    assert amfa
    assert amfa.accepts("aa")


def test_rpqs():
    graph = MultiDiGraph()
    graph.add_node(0, start_node=True, final_node=True)
    graph.add_edges_from(
        [
            (0, 0, {"label": "a"}),
            (0, 1, {"label": "b"}),
            (1, 0, {"label": "b"}),
            (1, 1, {"label": "a"}),
        ]
    )
    regex = "a|(ba)*"

    assert ms_bfs_based_rpq(regex, graph, {0}, {0}) == tensor_based_rpq(
        regex, graph, {0}, {0}
    )
    assert ms_bfs_based_rpq(regex, graph, {0}, {0}) == {(0, 0)}
