from pyformlang.finite_automaton import (
    NondeterministicFiniteAutomaton,
    DeterministicFiniteAutomaton,
)
from scipy.sparse import lil_array

from project.adjacency_matrix_fa import AdjacencyMatrixFA, intersect_automata
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
    assert amfa.accepts("abcd")


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
    dfa = intersect_automata(dfa1, dfa2)

    assert dfa
    assert dfa.accepts("aa")
