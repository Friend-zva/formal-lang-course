from dataclasses import dataclass
from typing import Set, Tuple, Dict

from networkx import DiGraph

from pyformlang.finite_automaton.finite_automaton import (
    State,
    Symbol,
    to_state,
    to_symbol,
)

from pyformlang.rsa import RecursiveAutomaton

from project.utils import rsm_to_nfa

RSMState = Tuple[Symbol, State]  # using in rsm_to_nfa


@dataclass(frozen=True)
class GSSNode:
    state_rsm: RSMState
    node_graph: int


class GSStack:
    # Graph Structured Stack
    def __init__(self):
        self._id_max: int = 0
        self._nodes: Dict[int, GSSNode] = {}
        self._nodes_lookup: Dict[GSSNode, int] = {}
        self._edges: Set[Tuple[int, RSMState, int]] = set()

    def add_node(self, item: GSSNode) -> int:
        id = self._nodes_lookup.get(item)
        if id:
            return id

        self._id_max += 1
        self._nodes[self._id_max] = item
        self._nodes_lookup[item] = self._id_max
        return self._id_max

    def add_edge(self, src: int, address_return: RSMState, dst: int) -> Tuple:
        edge = (src, address_return, dst)
        self._edges.add(edge)
        return edge

    def get_node(self, id: int) -> GSSNode | None:
        return self._nodes.get(id)

    @property
    def nodes(self):
        return self._nodes.copy()

    @property
    def edges(self):
        return self._edges.copy()


@dataclass(frozen=True)
class Descriptor:
    node_graph: int
    state_rsm: RSMState
    id_node_gss: int


log = 0


def gll_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: DiGraph,
    start_nodes: Set[int] = None,
    final_nodes: Set[int] = None,
) -> Set[Tuple[int, int]]:
    if start_nodes is None:
        start_nodes = set(graph.nodes)

    nfa_rsm = rsm_to_nfa(rsm)
    start_state: RSMState = (to_symbol("S'"), to_state(-1))
    final_state: RSMState = (to_symbol("S'"), to_state(-2))
    nfa_rsm.add_start_state(start_state)
    nfa_rsm.add_final_state(final_state)
    nfa_rsm.add_transition(start_state, rsm.initial_label, final_state)
    transitions_rsm: Dict = nfa_rsm.to_dict()
    if log:
        print("\nDict", transitions_rsm)
        print("Start", nfa_rsm.start_states)
        print("Final", nfa_rsm.final_states)

    stack = GSStack()
    process: Set[Descriptor] = set()

    for node in start_nodes:
        id = stack.add_node(GSSNode(start_state, node))
        process.add(Descriptor(node, start_state, id))

    result = set()
    processed: Set[Descriptor] = set()

    while process:
        if log == 2:
            print("\nProcess", process)
            print("Stack", stack.edges, stack.nodes)
        desc = process.pop()
        state_from = to_state(desc.state_rsm)

        processing: Set[Descriptor] = set()
        recall: Set[int] = set()

        if state_from in nfa_rsm.final_states:
            node_gss = stack.get_node(desc.id_node_gss)
            if desc.state_rsm == final_state:
                result.add((node_gss.node_graph, desc.node_graph))
            else:
                for src, sym, dst in stack.edges:
                    if src == desc.id_node_gss:
                        processing.add(Descriptor(desc.node_graph, sym, dst))

        if log == 2:
            print("Process", process)
            print("Stack", stack.edges, stack.nodes)
            print("Result", result)

        paths: Dict = transitions_rsm.get(state_from, {})
        if log == 1:
            print(f"\nDesc: {desc}")
            print(f"Paths: {paths}")
        for sym_rsm, states_to in paths.items():
            if sym_rsm in rsm.boxes:
                for state in nfa_rsm.start_states:
                    value = state.value
                    if sym_rsm.value != value[0]:
                        continue
                    id = stack.add_node(GSSNode(value, desc.node_graph))
                    processing.add(Descriptor(desc.node_graph, value, id))
                    cur_len = len(stack.edges)
                    for state_to in states_to:
                        stack.add_edge(id, state_to.value, desc.id_node_gss)
                        if log == 1:
                            print(id, state_to.value, desc.id_node_gss)
                    if cur_len != len(stack.edges):
                        recall.add(id)
                        for src, _, dst in stack.edges:
                            if dst == id:
                                node = stack.get_node(src)
                                processing.add(
                                    Descriptor(node.node_graph, node.state_rsm, src)
                                )
                                recall.add(src)
            else:
                for node_from, node_to, sym_g in graph.edges(data="label"):
                    if node_from == desc.node_graph and sym_rsm.value == sym_g:
                        for state_to in states_to:
                            processing.add(
                                Descriptor(node_to, state_to.value, desc.id_node_gss)
                            )
        if log:
            print("Processing", processing)

        if recall:
            processed = {item for item in processed if item.id_node_gss not in recall}

        for item in processing:
            if item not in processed or item.id_node_gss in recall:
                process.add(item)
                processed.add(item)

        if log == 3:
            print(f"Process: {process}")

    if log:
        print(stack, stack.edges, stack.nodes)

    if final_nodes is None:
        return result

    pairs = set()
    for s, f in result:
        if f in final_nodes:
            pairs.add((s, f))
    return pairs
