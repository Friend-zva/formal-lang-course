from pathlib import Path
from project.utils import (
    get_metadata,
    build_graph_two_cycles,
    regex_to_dfa,
    graph_to_nfa,
)
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton

import networkx as nx


PATH_GRAPHS = Path(__file__).parent / "graphs"


# Tests for "Task 1"


def test_get_metadata():
    metadata = get_metadata("generations")
    assert metadata.count_nodes == 129
    assert metadata.count_edges == 273
    assert metadata.tags_edges == {
        "rest",
        "type",
        "first",
        "onProperty",
        "intersectionOf",
        "equivalentClass",
        "hasValue",
        "hasChild",
        "hasParent",
        "inverseOf",
        "sameAs",
        "hasSibling",
        "oneOf",
        "someValuesFrom",
        "hasSex",
        "range",
        "versionInfo",
    }


def test_build_graph_two_cycles(tmp_path: Path):
    name_file = "two_cycles.dot"
    build_graph_two_cycles(4, 6, ("a", "b"), tmp_path / name_file)
    assert nx.utils.graphs_equal(
        nx.nx_pydot.read_dot(tmp_path / name_file),
        nx.nx_pydot.read_dot(PATH_GRAPHS / "source" / name_file),
    )


# Tests for "Task 2"


def test_regex_to_dfa():
    dfa = regex_to_dfa("a(b)*(c|d)")

    assert not dfa.is_empty()
    assert dfa.is_deterministic()
    assert dfa.is_equivalent_to(dfa.minimize())
    assert dfa.accepts("abbbd")


def test_graph_to_nfa_from_usual_graph():
    edges = [(1, 2), (2, 3), (3, 4)]
    graph = nx.MultiDiGraph(edges)
    nfa = graph_to_nfa(graph, [], [])

    assert not nfa.is_empty()
    assert nfa.start_states == {1, 2, 3, 4}
    assert nfa.final_states == {1, 2, 3, 4}


def test_graph_to_nfa_from_nfa():
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

    graph = nfa.to_networkx()
    nfa_from = graph_to_nfa(graph, [start_state], [final_state])

    assert not nfa_from.is_empty()
    assert nfa_from.is_equivalent_to(nfa)
    assert nfa_from.accepts("aabcb")
