from symtable import Symbol
from typing import Iterable, Set, Dict

from pyformlang.finite_automaton.finite_automaton import to_state, to_symbol
from scipy.sparse import lil_array

from pyformlang.finite_automaton import (
    NondeterministicFiniteAutomaton, State,
)


class AdjacencyMatrixFA:
    def __init__(self, automation: NondeterministicFiniteAutomaton = None):
        if automation:
            self._init_from_automation(automation)
        else:
            self._init_empty()

    def _init_empty(self):
        self._states_ids = dict()
        self._symbols = set()
        self._adjacency_matrices = dict()
        self._start_states = set()
        self._final_states = set()

    def _init_from_automation(self, fa: NondeterministicFiniteAutomaton):
        self._states_ids = self._gen_states_ids(fa.states)
        self._symbols = fa.symbols
        self._adjacency_matrices = self._gen_empty_adjacency_matrices()
        self._start_states = fa.start_states
        self._final_states = fa.final_states

        graph = fa.to_networkx()
        for src, dst, sbl in graph.edges(data="label"):
            if sbl is None: continue
            symbol = to_symbol(sbl)
            source_state = self._get_state_id(src)
            destination_state = self._get_state_id(dst)
            self._adjacency_matrices[symbol][source_state, destination_state] = True

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

    @property
    def _states(self) -> Set[State]:
        return set(self._states_ids.keys())

    def _gen_states_ids(self, states: Set[State]) -> dict[State, int]:
        return dict((to_state(st), id) for id, st in enumerate(states))

    def _get_state_id(self, state: State) -> int:
        state = to_state(state)
        return self._states_ids[state]

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
