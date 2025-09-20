from symtable import Symbol
from typing import Iterable, Set, Dict

from pyformlang.finite_automaton.finite_automaton import to_state, to_symbol
from scipy.sparse import lil_array

from pyformlang.finite_automaton import (
    NondeterministicFiniteAutomaton, State,
)


class AdjacencyMatrixFA:
    def __init__(
        self,
        states: Set[State] = None,
        symbols: Set[Symbol] = None,
        adjacency_matrices: Dict[Symbol, lil_array] = None,
        start_states: Set[State] = None,
        final_states: Set[State] = None,
    ):
        if states:
            states = {to_state(st) for st in states}
        self._states = states or set()

        if symbols:
            symbols = {to_symbol(sym) for sym in symbols}
        self._symbols = symbols or set()

        self._adjacency_matrices = adjacency_matrices or self._gen_empty_adjacency_matrices()

        if start_states:
            start_states = {to_state(st) for st in start_states}
        self._start_states = start_states or set()

        if final_states:
            final_states = {to_state(st) for st in final_states}
        self._final_states = final_states or set()

        for state in self._start_states:
            if state and state not in self._states:
                self._states.add(state)

        for state in self._final_states:
            if state and state not in self._states:
                self._states.add(state)

        self._state_ids = self._gen_state_ids()

    def __init__(self, fa: NondeterministicFiniteAutomaton):
        self._states = fa.states
        self._symbols = fa.symbols
        self._adjacency_matrices = self._gen_empty_adjacency_matrices()
        self._start_states = fa.start_states
        self._final_states = fa.final_states
        self._state_ids = self._gen_state_ids()

        graph = fa.to_networkx()
        for src, dst, sbl in graph.edges(data="label"):
            if sbl is None: continue
            symbol = to_symbol(sbl)
            source_state = self._get_state_id(src)
            destination_state = self._get_state_id(dst)
            self._adjacency_matrices[symbol][source_state, destination_state] = True

        print(self._adjacency_matrices)

    def accepts(self, word: Iterable[Symbol]) -> bool:
        word = [to_symbol(x) for x in word]
        current_states = self._start_states
        for symbol in word:
            current_states = self._get_next_states(
                current_states,
                symbol
            )
        return any(self.is_final_state(st) for st in current_states)

    def is_empty(self) -> bool:
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
                for state in self._transition_function(state, symbol):
                    if state not in processed:
                        to_process.append(state)
                        processed.add(state)

        return True

    def is_final_state(self, state: State) -> bool:
        state = to_state(state)
        return state in self._final_states

    def _gen_state_ids(self) -> dict[State, int]:
        return dict((st, id) for id, st in enumerate(self._states))

    def _get_state_id(self, state: State) -> int:
        state = to_state(state)
        return self._state_ids[state]

    def _gen_empty_adjacency_matrices(self) -> dict[Symbol, lil_array]:
        measure = len(self._states)
        matrix = lil_array((measure, measure), dtype=bool)
        return dict((symbol, matrix) for symbol in self._symbols)

    def _get_next_states(
        self,
        states: Iterable[State],
        symbol: Symbol
    ) -> Set[State]:
        next_states = set()
        for state in states:
            for n_state in self._transition_function(state, symbol):
                next_states.add(n_state)

        return next_states

    def _transition_function(self, state, symbol) -> Set[State]:
        next_states = set()
        state = self._get_state_id(state)
        for n_state in self._states:
            next_state = self._get_state_id(n_state)
            if self._adjacency_matrices[symbol][state, next_state]:
                next_states.add(n_state)

        return next_states
