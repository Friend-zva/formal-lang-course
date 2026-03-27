from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Any, Set

import cfpq_data as cd
import networkx as nx

from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
)
from pyformlang.finite_automaton.finite_automaton import to_state, to_symbol
from pyformlang.regular_expression import Regex

from pyformlang.rsa import RecursiveAutomaton
from pyformlang.cfg import CFG, Production, Epsilon


@dataclass
class MetadataGraph:
    count_nodes: int
    count_edges: int
    tags_edges: Set[Any]


def get_metadata(name_graph: str) -> MetadataGraph:
    """Loads graph metadata from dataset

    Parameters
    ----------
    name_graph : str
        The name of the graph from the dataset

    Returns
    -------
    metadata : MetadataGraph
        Graph metadata
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
    Saves it into DOT file with given path

    Parameters
    ----------
    n : int
        The number of nodes in the first cycle without a common node

    m : int
        The number of nodes in the second cycle without a common node

    labels : Tuple[str, str]
        Labels that will be used to mark the edges of the graph

    path_save : str or file
        Filename or file handle for saving
    """
    graph = cd.labeled_two_cycles_graph(n, m, labels=labels)
    nx.drawing.nx_pydot.write_dot(graph, path_save)


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    """Transforms the regular expression into DFA

    Parameters
    ----------
    regex : str
        The regex represented as a string

    Returns
    ----------
    dfa : :class:`~pyformlang.finite_automaton.DeterministicFiniteAutomaton`
        Deterministic Finite Automaton equivalent to the regex
    """
    reg = Regex(regex)
    enfa = reg.to_epsilon_nfa()
    dfa = enfa.to_deterministic()
    return dfa.minimize()


def graph_to_nfa(
    graph: nx.MultiDiGraph, start_states: Set[int], final_states: Set[int]
) -> NondeterministicFiniteAutomaton:
    """Imports a networkx graph into NFA. Adds new initial and final states

    Parameters
    ----------
    graph : :class:`~networkx.MultiDiGraph`
        The graph representation of the automaton

    start_states : Set[int]
        New initial states

    final_states : Set[int]
        New final states

    Returns
    ----------
    nfa : :class:`~pyformlang.finite_automaton.NondeterministicFiniteAutomaton`
        Nondeterministic Finite Automaton read from the graph
    """
    nfa = NondeterministicFiniteAutomaton.from_networkx(graph)

    for state in start_states or graph.nodes:
        nfa.add_start_state(to_state(state))

    for state in final_states or graph.nodes:
        nfa.add_final_state(to_state(state))

    return nfa


def cfg_to_weak_normal_form(cfg: CFG) -> CFG:
    """Gets the Chomsky Weakened Normal Form of a Context Free Grammar

    Parameters
    ----------
    cfg : :class:`~pyformlang.cfg.CFG`
        An original Context-Free Grammar

    Returns
    -------
    cwnf : :class:`~pyformlang.cfg.CFG`
        A new CFG equivalent in the Context-Weak-Free Grammar
    """
    cfg_nf = cfg.to_normal_form()

    prods_eps = set(cfg_nf.productions)

    for var in cfg.get_nullable_symbols():
        prods_eps.add(Production(var, [Epsilon()]))

    cwnf = CFG(
        variables=cfg_nf.variables,
        terminals=cfg_nf.terminals,
        start_symbol=cfg_nf.start_symbol,
        productions=prods_eps,
    )
    return cwnf.remove_useless_symbols()


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
