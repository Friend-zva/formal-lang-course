from symtable import Symbol
from typing import Iterable, Set, Dict

from pyformlang.finite_automaton.finite_automaton import to_state, to_symbol
from scipy.sparse import lil_array, kron, csr_array, eye_array

from pyformlang.finite_automaton import (
    NondeterministicFiniteAutomaton,
    State,
)

import networkx as nx

from project.utils import graph_to_nfa, regex_to_dfa


def gen_states_ids(states: Set[State]) -> Dict[State, int]:
    return dict((st, id) for id, st in enumerate(states))


def gen_empty_adjacency_matrices(
    count_states, symbols, is_compressed
) -> Dict[Symbol, lil_array | csr_array]:
    def create_matrix():
        if is_compressed:
            return csr_array((count_states, count_states), dtype=bool)
        else:
            return lil_array((count_states, count_states), dtype=bool)

    return dict((to_symbol(sym), create_matrix()) for sym in symbols)


class AdjacencyMatrixFA:
    def __init__(
        self,
        automation: NondeterministicFiniteAutomaton = None,
        states_ids: Dict[State, int] = None,
        symbols: Set[Symbol] = None,
        adjacency_matrices: Dict[Symbol, lil_array | csr_array] = None,
        start_states: Set[State] = None,
        final_states: Set[State] = None,
    ):
        if automation:
            self._init_from_automation(automation)
        else:
            self._init_from_arguments(
                states_ids, symbols, adjacency_matrices, start_states, final_states
            )

    def _init_from_arguments(
        self,
        states_ids: Dict[State, int] = None,
        symbols: Set[Symbol] = None,
        adjacency_matrices: Dict[Symbol, lil_array | csr_array] = None,
        start_states: Set[State] = None,
        final_states: Set[State] = None,
    ):
        if states_ids:
            states_ids = dict((to_state(st), id) for (st, id) in states_ids.items())
        self._states_ids = states_ids or dict()

        if symbols:
            symbols = {to_symbol(sym) for sym in symbols}
        self._symbols = symbols or set()

        if adjacency_matrices:
            adjacency_matrices = dict(
                (to_symbol(sym), arr) for (sym, arr) in adjacency_matrices.items()
            )
        self._adjacency_matrices = adjacency_matrices or gen_empty_adjacency_matrices(
            self.count_states, self._symbols, is_compressed=True
        )

        if start_states:
            for state in start_states:
                if state not in self.states:
                    start_states.remove(state)
            start_states = {to_state(st) for st in start_states}
        self._start_states = start_states or set()

        if final_states:
            for state in final_states:
                if state not in self.states:
                    final_states.remove(state)
            final_states = {to_state(st) for st in final_states}
        self._final_states = final_states or set()

    def _init_from_automation(self, fa: NondeterministicFiniteAutomaton):
        self._states_ids = gen_states_ids(fa.states)
        self._symbols = fa.symbols
        self._adjacency_matrices: Dict[Symbol, lil_array] = (
            gen_empty_adjacency_matrices(
                self.count_states, self._symbols, is_compressed=False
            )
        )
        self._start_states = fa.start_states
        self._final_states = fa.final_states

        graph = fa.to_networkx()
        for src, dst, sym in graph.edges(data="label"):
            if sym:
                self.add_transition(src, sym, dst)

        self._adjacency_matrices = dict(
            (sym, m.tocsr()) for (sym, m) in self._adjacency_matrices.items()
        )

    def accepts(self, word: Iterable[Symbol]) -> bool:
        word = [to_symbol(x) for x in word]
        states = self._start_states
        for symbol in word:
            states = self._get_next_states(states, symbol)
        return any(self.is_final_state(st) for st in states)

    def is_empty(self) -> bool:
        return self.is_empty_transition_closure()

    def is_empty_processing(self) -> bool:
        to_process = []
        processed = set()
        for state in self._start_states:
            to_process.append(state)
            processed.add(state)
        while to_process:
            state = to_process.pop()
            if self.is_final_state(state):
                return False
            for symbol in self._symbols:
                for n_state in self._transition_function(state, symbol):
                    if n_state not in processed:
                        to_process.append(n_state)
                        processed.add(n_state)

        return True

    def is_empty_transition_closure(self) -> bool:
        ts_matrix = self.transition_closure()

        for _, start_id in self.start_states_ids.items():
            for _, final_id in self.final_states_ids.items():
                if ts_matrix[start_id, final_id]:
                    return False

        return True

    def is_start_state(self, state: State) -> bool:
        state = to_state(state)
        return state in self._start_states

    def is_final_state(self, state: State) -> bool:
        state = to_state(state)
        return state in self._final_states

    @property
    def states_ids(self) -> Dict[State, int]:
        return self._states_ids.copy()

    @property
    def states(self) -> Set[State]:
        return set(self._states_ids.keys())

    @property
    def symbols(self) -> Set[Symbol]:
        return self._symbols.copy()

    @property
    def count_states(self) -> int:
        return len(self._states_ids)

    @property
    def start_states(self) -> Set[State]:
        return self._start_states.copy()

    @property
    def start_states_ids(self) -> Dict[State, int]:
        return dict(
            (st, id) for (st, id) in self._states_ids.items() if self.is_start_state(st)
        )

    @property
    def final_states(self) -> Set[State]:
        return self._final_states.copy()

    @property
    def final_states_ids(self) -> Dict[State, int]:
        return dict(
            (st, id) for (st, id) in self._states_ids.items() if self.is_final_state(st)
        )

    @property
    def adjacency_matrices(self) -> Dict[Symbol, lil_array | csr_array]:
        return self._adjacency_matrices.copy()

    def get_state_id(self, state: State) -> int:
        state = to_state(state)
        return self._states_ids[state]

    def add_state_id(self, state: State, id: int):
        state = to_state(state)
        self._states_ids[state] = id

    def add_start_state(self, state: State) -> bool:
        state = to_state(state)
        if state in self.states:
            self._start_states.add(state)
            return True
        else:
            return False

    def add_final_state(self, state: State) -> bool:
        state = to_state(state)
        if state in self.states:
            self._final_states.add(state)
            return True
        else:
            return False

    def is_transition(self, state_from: State, symbol: Symbol, state_to: State) -> bool:
        try:
            state_from = self.get_state_id(state_from)
            symbol = to_symbol(symbol)
            state_to = self.get_state_id(state_to)
            return self._adjacency_matrices[symbol][state_from, state_to]
        except:
            return False

    def add_transition(self, state_from: State, symbol: Symbol, state_to: State):
        try:
            state_from = self.get_state_id(state_from)
            symbol = to_symbol(symbol)
            state_to = self.get_state_id(state_to)
            self._adjacency_matrices[symbol][state_from, state_to] = True
        except:
            return

    def transition_closure(self) -> csr_array:
        max_count_non_zero = self.count_states * self.count_states
        e_matrix = eye_array(self.count_states, dtype=bool, format="csr")

        tc_matrix = e_matrix
        for symbol in self._symbols:
            tc_matrix += self._adjacency_matrices[symbol]

        for _ in range(0, self.count_states):
            count_non_zero = tc_matrix.nnz
            if count_non_zero == max_count_non_zero:
                break

            tc_matrix @= tc_matrix

        return tc_matrix

    def _get_next_states(self, states: Iterable[State], symbol: Symbol) -> Set[State]:
        next_states = set()
        for state in states:
            for n_state in self._transition_function(state, symbol):
                next_states.add(n_state)

        return next_states

    def _transition_function(self, state: State, symbol: Symbol) -> Set[State]:
        next_states = set()
        for n_state in self.states:
            if self.is_transition(state, symbol, n_state):
                next_states.add(n_state)

        return next_states


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    states_ids, adjacency_matrices = dict(), dict()
    start_states, final_states = set(), set()
    symbols = automaton1.symbols.intersection(automaton2.symbols)

    for symbol in symbols:
        adjacency_matrices[symbol] = kron(
            automaton1.adjacency_matrices[symbol],
            automaton2.adjacency_matrices[symbol],
            format="csr",
        )

    for state1, id1 in automaton1.states_ids.items():
        for state2, id2 in automaton2.states_ids.items():
            id = id1 * automaton2.count_states + id2
            state = to_state(id)
            states_ids[state] = id

            if automaton1.is_start_state(state1) and automaton2.is_start_state(state2):
                start_states.add(state)

            if automaton1.is_final_state(state1) and automaton2.is_final_state(state2):
                final_states.add(state)

    return AdjacencyMatrixFA(
        states_ids=states_ids,
        symbols=symbols,
        start_states=start_states,
        final_states=final_states,
        adjacency_matrices=adjacency_matrices,
    )


def tensor_based_rpq(
    regex: str, graph: nx.MultiDiGraph, start_nodes: Set[int], final_nodes: Set[int]
) -> set[tuple[int, int]]:
    nfa = graph_to_nfa(graph, start_nodes, final_nodes)
    graph_amfa = AdjacencyMatrixFA(nfa)

    dfa = regex_to_dfa(regex)
    request_amfa = AdjacencyMatrixFA(dfa)

    front_amfa = intersect_automata(graph_amfa, request_amfa)
    tc_front = front_amfa.transition_closure()

    fronts = set()
    for s_st_graph, s_id_graph in graph_amfa.start_states_ids.items():
        for f_st_graph, f_id_graph in graph_amfa.final_states_ids.items():
            for _, s_id_request in request_amfa.start_states_ids.items():
                for _, f_id_request in request_amfa.final_states_ids.items():
                    if tc_front[
                        s_id_graph * request_amfa.count_states + s_id_request,
                        f_id_graph * request_amfa.count_states + f_id_request,
                    ]:
                        fronts.add((s_st_graph.value, f_st_graph.value))

    return fronts
