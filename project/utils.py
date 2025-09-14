from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Any, Set

import cfpq_data as cd
import networkx as nx

from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
    State,
)
from pyformlang.regular_expression import Regex


@dataclass
class MetadataGraph:
    count_nodes: int
    count_edges: int
    tags_edges: Set[Any]


def get_metadata(name_graph: str) -> MetadataGraph:
    """Loads graph metadata from dataset.

    Parameters
    ----------
    name_graph : str
        The name of the graph from the dataset.

    Returns
    -------
    metadata : MetadataGraph
        Graph metadata.
    """
    archive = cd.download(name_graph)
    graph = cd.graph_from_csv(archive)

    return MetadataGraph(
        graph.number_of_nodes(),
        graph.number_of_edges(),
        set(cd.get_sorted_labels(graph)),
    )


def build_graph_two_cycles(n: int, m: int, labels: Tuple[str, str], path_save: Path):
    """Builds a graph with two cycles connected by one node. With labeled edges.
    Saves it into DOT file with given path.

    Parameters
    ----------
    n : int
        The number of nodes in the first cycle without a common node.

    m : int
        The number of nodes in the second cycle without a common node.

    labels : Tuple[str, str]
        Labels that will be used to mark the edges of the graph.

    path_save : str or file
        Filename or file handle for saving.
    """
    graph = cd.labeled_two_cycles_graph(n, m, labels=labels)
    nx.drawing.nx_pydot.write_dot(graph, path_save)


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    """Transforms the regular expression into a dfa.

    Parameters
    ----------
    regex : str
        The regex represented as a string.

    Returns
    ----------
    dfa : pyformlang.finite_automaton.DeterministicFiniteAutomaton
        A dfa equivalent to the regex.
    """
    reg = Regex(regex)
    enfa = reg.to_epsilon_nfa()
    return enfa.to_deterministic()


def graph_to_nfa(
    graph: nx.MultiDiGraph, start_states: Set[int], final_states: Set[int]
) -> NondeterministicFiniteAutomaton:
    """Imports a networkx graph into an nfa. Adds new initial and final states.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The graph representation of the automaton.

    start_states : Set[int]
        New initial states.

    final_states : Set[int]
        New final states.

    Returns
    ----------
    nfa : pyformlang.finite_automaton.NondeterministicFiniteAutomaton
        A nfa read from the graph.
    """
    nfa = NondeterministicFiniteAutomaton.from_networkx(graph)

    for state in start_states or graph.nodes:
        nfa.add_start_state(State(state))

    for state in final_states or graph.nodes:
        nfa.add_final_state(State(state))

    return nfa
