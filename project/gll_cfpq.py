from dataclasses import dataclass
from typing import Set, Tuple, Dict
from collections import deque

from networkx import DiGraph

from pyformlang.finite_automaton.finite_automaton import (
    State,
    Symbol,
    to_state,
    to_symbol,
)

from pyformlang.rsa import RecursiveAutomaton

from project.utils import rsm_to_nfa

RSMState = Tuple[Symbol, State]
"""Type alias for RSM state representation

Also used in the function `~project.utils.rsm_to_nfa`.

Parameters
----------
Symbol : type
    The nonterminal symbol from the grammar
State  : type
    The state of the corresponding nonterminal in the automaton
"""


@dataclass(frozen=True)
class GSSNode:
    """A node in GSS (Graph Structured Stack)

    Parameters
    ----------
    state_rsm  : RSMState
        The current state in the Recursive State Machine
    node_graph : int
        The node in the input graph
    """

    state_rsm: RSMState
    node_graph: int


class GSStack:
    """Represent Graph Structured Stack for GLL-based context-free path querying"""

    def __init__(self):
        self._id_max: int = 0
        self._nodes: Dict[int, GSSNode] = {}
        self._nodes_lookup: Dict[GSSNode, int] = {}
        self._edges: Set[Tuple[int, RSMState, int]] = set()

    def add_node(self, item: GSSNode) -> int:
        """Add a GSS node to the stack or return existing node id

        Parameters
        ----------
        item : GSSNode
            The GSS node to add to the stack

        Returns
        -------
        node_id : int
            The node id (either newly created or existing)
        """
        id = self._nodes_lookup.get(item)
        if id:
            return id

        self._id_max += 1
        self._nodes[self._id_max] = item
        self._nodes_lookup[item] = self._id_max
        return self._id_max

    def add_edge(self, src: int, address_return: RSMState, dst: int) -> Tuple:
        """Add a directed edge to the GSS

        Parameters
        ----------
        src : int
            The source GSS node id
        address_return : RSMState
            The return address (RSM state) as a label
        dst : int
            The destination GSS node id

        Returns
        -------
        edge : Tuple[int, RSMState, int]
            The added or existed edge
        """
        edge = (src, address_return, dst)
        self._edges.add(edge)
        return edge

    def get_node(self, id: int) -> GSSNode | None:
        """Get a GSS node by its id

        Parameters
        ----------
        id : int
            The node id

        Returns
        -------
        node : GSSNode or None
            The GSS node with the given id, or None if no such node exists
        """
        return self._nodes.get(id)

    @property
    def edges(self) -> Set[Tuple]:
        """Get all edges in the GSS

        Returns
        -------
        edges : Set[Tuple[int, RSMState, int]]
            A copy of the set of all edges in the GSS
        """
        return self._edges.copy()

    @property
    def count_edges(self) -> int:
        """Get count of all edges in the GSS

        Returns
        -------
        count : int
            Count of edges
        """
        return len(self._edges)


@dataclass(frozen=True)
class Descriptor:
    """A parsing state, which is a single GLL step

    Parameters
    ----------
    node_graph  : int
        The last processed node in the input graph
    state_rsm   : RSMState
        The current RSM state
    id_node_gss : int
        The GSS node id representing the context of the start of parsing
    """

    node_graph: int
    state_rsm: RSMState
    id_node_gss: int


class Process:
    """Manage processing descriptors with LIFO order and uniqueness

    Maintains three synchronized data structures:
    - `process`: LIFO deque for descriptor processing order
    - `_process`: Set for existence checks in processing queue and provides uniqueness
    - `_processed`: Set for tracking already processed descriptors
    """

    def __init__(self):
        self._id_max: int = 0
        self.process: deque[Descriptor] = deque()
        self._process: Set[Descriptor] = set()
        self._processed: Set[Descriptor] = set()

    def push(self, desc: Descriptor, processed=False):
        """Add descriptor to processing queue

        Parameters
        ----------
        desc : Descriptor
            Descriptor to add for processing
        processed : bool, default=False
            If True, also marks descriptor as processed immediately
            Used for initial descriptors to prevent reprocessing
        """
        if not processed:
            self.process.append(desc)
            self._process.add(desc)
            return
        if desc not in self._processed and desc not in self._process:
            self.process.append(desc)
            self._process.add(desc)
            self._processed.add(desc)

    def pop(self) -> Descriptor:
        """Remove and return next descriptor from processing queue (LIFO)

        Returns
        -------
        desc : Descriptor
            Next descriptor for processing
        """
        desc = self.process.popleft()
        self._process.remove(desc)
        return desc

    def edit(self, recall: Set[int]):
        """Remove descriptors with specified GSS node ids from processed set

        Parameters
        ----------
        recall : Set[int]
            Set of GSS node ids
        """
        self._processed = {
            desc for desc in self._processed if desc.id_node_gss not in recall
        }

    @property
    def continues(self) -> bool:
        """Check if processing queue has more descriptors

        Returns
        -------
        cont : bool
            True if processing queue is not empty
        """
        return bool(self.process)


def gll_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: DiGraph,
    start_nodes: Set[int] = None,
    final_nodes: Set[int] = None,
) -> Set[Tuple[int, int]]:
    """Evaluate RSM path queries using Generalized LL algorithm (with GSS)

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

    nfa_rsm = rsm_to_nfa(rsm)
    start_state: RSMState = (to_symbol("S'"), to_state(-1))
    nfa_rsm.add_start_state(start_state)
    final_state: RSMState = (to_symbol("S'"), to_state(-2))
    nfa_rsm.add_final_state(final_state)
    nfa_rsm.add_transition(start_state, rsm.initial_label, final_state)
    transitions_rsm: Dict = nfa_rsm.to_dict()

    stack = GSStack()
    process = Process()

    for node in start_nodes:
        id = stack.add_node(GSSNode(start_state, node))
        desc = Descriptor(node, start_state, id)
        process.push(desc)

    result = set()
    while process.continues:
        desc = process.pop()
        state_from = to_state(desc.state_rsm)

        processing: deque[Descriptor] = deque()
        recall: Set[int] = set()

        if state_from in nfa_rsm.final_states:
            node_gss = stack.get_node(desc.id_node_gss)
            if desc.state_rsm == final_state:
                result.add((node_gss.node_graph, desc.node_graph))
                continue
            else:
                for src, sym, dst in stack.edges:
                    if src == desc.id_node_gss:
                        processing.append(Descriptor(desc.node_graph, sym, dst))

        paths: Dict = transitions_rsm.get(state_from, {})
        for sym_rsm, states_to in paths.items():
            if sym_rsm in rsm.boxes:
                for state in nfa_rsm.start_states:
                    value = state.value
                    if value[0] != sym_rsm.value:
                        continue
                    id = stack.add_node(GSSNode(value, desc.node_graph))
                    processing.append(Descriptor(desc.node_graph, value, id))
                    count = stack.count_edges
                    for state_to in states_to:
                        stack.add_edge(id, state_to.value, desc.id_node_gss)
                    if count == stack.count_edges:
                        continue
                    recall.add(id)
                    recall.add(desc.id_node_gss)
                    for src, _, dst in stack.edges:
                        if dst != id:
                            continue
                        node = stack.get_node(src)
                        processing.append(
                            Descriptor(node.node_graph, node.state_rsm, src)
                        )
                        recall.add(src)
            else:
                for node_from, node_to, sym_g in graph.edges(data="label"):
                    if node_from == desc.node_graph and sym_rsm.value == sym_g:
                        for state_to in states_to:
                            processing.append(
                                Descriptor(node_to, state_to.value, desc.id_node_gss)
                            )

        if recall:
            process.edit(recall)

        for desc in processing:
            process.push(desc, processed=True)

    if final_nodes is None:
        return result

    pairs = set()
    for s, f in result:
        if f in final_nodes:
            pairs.add((s, f))
    return pairs
