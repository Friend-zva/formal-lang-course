from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, DeterministicFiniteAutomaton

from project.adjacency_matrix_fa import AdjacencyMatrixFA


def test_adjacency_matrix_fa_empty():
    amfa = AdjacencyMatrixFA()
    assert amfa
    assert not amfa.accepts("a")
    assert amfa.is_empty()


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
