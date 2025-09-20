from pyformlang.finite_automaton import NondeterministicFiniteAutomaton

from project.adjacency_matrix_fa import AdjacencyMatrixFA


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
    assert amfa.accepts("aabcb")
    assert not amfa.is_empty()
