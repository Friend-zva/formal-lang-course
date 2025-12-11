from typing import Iterable, Set, Dict, Tuple, Any

from networkx import MultiDiGraph

from scipy.sparse import lil_array, kron, csr_array, eye_array

from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, State, Symbol
from pyformlang.finite_automaton.finite_automaton import to_state, to_symbol

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
    """Represents a finite automaton using adjacency matrices

    Parameters
    ----------
    automation : :class:`~pyformlang.finite_automaton.NondeterministicFiniteAutomaton`, optional
        An existing finite automaton
    states_ids : dict of :class:`~pyformlang.finite_automaton.State` to int, optional
        The mapping of states to ids
    symbols : set of :class:`~pyformlang.finite_automaton.Symbol`, optional
        A set of symbols
    adjacency_matrices : dict of :class:`~pyformlang.finite_automaton.Symbol` to \
    :class:`~scipy.sparse.lil_array` or :class:`~scipy.sparse.csr_array`, optional
        Sparse adjacency matrices for each symbol representing transitions
    start_states : set of :class:`~pyformlang.finite_automaton.State`, optional
        A set of start states
    final_states : set of :class:`~pyformlang.finite_automaton.State`, optional
        A set of final states
    """

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
        self._ids_states_cache: Dict[int, State] | None = None

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
                src = to_state(src)
                dst = to_state(dst)
                sym = to_symbol(sym)
                self.add_transition(src, sym, dst)

        self._adjacency_matrices = dict(
            (sym, m.tocsr()) for (sym, m) in self._adjacency_matrices.items()
        )

    def accepts(self, word: Iterable[Any]) -> bool:
        """Checks whether the AMFA accepts a given word

        Parameters
        ----------
        word : iterable of Any
            The input word to process

        Returns
        -------
        is_accepted : bool
            Whether the word is accepted or not
        """
        word = [to_symbol(x) for x in word]
        states = self._start_states
        for symbol in word:
            states = self._get_next_states(states, symbol)
        return any(self.is_final_state(st) for st in states)

    def is_empty(self) -> bool:
        """Checks if the language represented by the AMFA is empty or not

        Returns
        ----------
        is_empty : bool
            Whether the language is empty or not
        """
        return self.is_empty_transition_closure()

    def is_empty_processing(self) -> bool:
        """Check emptiness using state processing (BFS)

        Returns
        -------
        is_empty : bool
            Whether the language is empty or not
        """
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
        """Check emptiness using transitive closure matrix

        Returns
        -------
        is_empty : bool
            Whether the language is empty or not
        """
        ts_matrix = self.transition_closure()

        for _, s_id in self.start_states_ids.items():
            for _, f_id in self.final_states_ids.items():
                if ts_matrix[s_id, f_id]:
                    return False

        return True

    def is_start_state(self, state: Any) -> bool:
        """Checks if a state is start

        Parameters
        ----------
        state : Any
            The state to check

        Returns
        -------
        is_start : bool
            Whether the state is start or not
        """
        state = to_state(state)
        return state in self._start_states

    def is_final_state(self, state: Any) -> bool:
        """Checks if a state is final

        Parameters
        ----------
        state : Any
            The state to check

        Returns
        -------
        is_final : bool
            Whether the state is final or not
        """
        state = to_state(state)
        return state in self._final_states

    @property
    def states_ids(self) -> Dict[State, int]:
        """The mapping of states to ids"""
        return self._states_ids.copy()

    @property
    def states(self) -> Set[State]:
        """All states"""
        return set(self._states_ids.keys())

    @property
    def symbols(self) -> Set[Symbol]:
        """The symbols"""
        return self._symbols.copy()

    @property
    def count_states(self) -> int:
        """The number of all states"""
        return len(self._states_ids)

    @property
    def start_states(self) -> Set[State]:
        """The start states"""
        return self._start_states.copy()

    @property
    def start_states_ids(self) -> Dict[State, int]:
        """The mapping of start states to ids"""
        return dict(
            (st, id) for (st, id) in self._states_ids.items() if self.is_start_state(st)
        )

    @property
    def final_states(self) -> Set[State]:
        """The final states"""
        return self._final_states.copy()

    @property
    def final_states_ids(self) -> Dict[State, int]:
        """The mapping of final states to ids"""
        return dict(
            (st, id) for (st, id) in self._states_ids.items() if self.is_final_state(st)
        )

    @property
    def adjacency_matrices(self) -> Dict[Symbol, lil_array | csr_array]:
        """The mapping of adjacency matrices to all symbols"""
        return self._adjacency_matrices.copy()

    def get_state_by_id(self, id: int) -> State | None:
        """Get the state for an id

        Parameters
        ----------
        id : Int
            The id to look up

        Returns
        -------
        state : int | None
            The state if found, None otherwise
        """
        if self._ids_states_cache is None:
            self._ids_states_cache = {id: st for st, id in self.states_ids.items()}
        return self._ids_states_cache.get(id)

    def get_id_by_state(self, state: Any) -> int | None:
        """Get the integer id for a state

        Parameters
        ----------
        state : Any
            The state to look up

        Returns
        -------
        state_id : int | None
            The state id if found, None otherwise
        """
        state = to_state(state)
        return self._states_ids.get(state)

    def add_state_id(self, state: Any, id: int):
        """Add a state with specified id

        Parameters
        ----------
        state : Any
            The state to add
        id : int
            The integer id
        """
        state = to_state(state)
        self._states_ids[state] = id
        self._ids_states_cache = None

    def add_start_state(self, state: Any) -> bool:
        """Add a start state

        Parameters
        ----------
        state : Any
            The state to mark as start

        Returns
        -------
        is_added : bool
            True if successful, False if state doesn't exist in set of states
        """
        state = to_state(state)
        if state in self.states:
            self._start_states.add(state)
            return True
        else:
            return False

    def add_final_state(self, state: Any) -> bool:
        """Add a final state

        Parameters
        ----------
        state : Any
            The state to mark as final

        Returns
        -------
        is_added : bool
            True if successful, False if state doesn't exist in set of states
        """
        state = to_state(state)
        if state in self.states:
            self._final_states.add(state)
            return True
        else:
            return False

    def add_symbol(self, symbol: Any) -> bool:
        """Add a new symbol

        Parameters
        ----------
        symbol : Any
            The symbol

        Returns
        -------
        is_added : bool
            True if successful, False if state already exist in set of symbols
        """
        symbol = to_symbol(symbol)
        if symbol not in self.symbols:
            self._symbols.add(symbol)
            self._adjacency_matrices[symbol] = lil_array(
                (self.count_states, self.count_states), dtype=bool
            )
            return True
        else:
            return False

    def is_transition(self, state_from: Any, symbol: Any, state_to: Any) -> bool:
        """Check if a transition exists

        Parameters
        ----------
        state_from : Any
            The source state
        symbol : Any
            The input symbol
        state_to : Any
            The destination state

        Returns
        -------
        is_transition : bool
            True if the transition exists
        """
        state_from = self.get_id_by_state(state_from)
        symbol = to_symbol(symbol)
        state_to = self.get_id_by_state(state_to)
        if state_from is not None and state_to is not None:
            return self._adjacency_matrices[symbol][state_from, state_to]
        else:
            return False

    def add_transition(self, state_from: Any, symbol: Any, state_to: Any):
        """Add a transition to the AMFA

        Parameters
        ----------
        state_from : Any
            The source state
        symbol : Any
            The input symbol
        state_to : Any
            The destination state
        """
        state_from = self.get_id_by_state(state_from)
        symbol = to_symbol(symbol)
        state_to = self.get_id_by_state(state_to)
        if state_from is not None and state_to is not None:
            self._adjacency_matrices[symbol][state_from, state_to] = True

    def transition_closure(self) -> csr_array:
        """Compute the transitive closure of transitions

        Returns
        -------
        tc_matrix : csr_array
            Boolean matrix where closure[i,j] indicates reachability from i to j
        """
        max_count_non_zero = self.count_states * self.count_states
        e_matrix = eye_array(self.count_states, dtype=bool, format="csr")

        tc_matrix = e_matrix
        for symbol in self._symbols:
            tc_matrix += self._adjacency_matrices[symbol]

        for _ in range(0, self.count_states):
            count_non_zero = tc_matrix.count_nonzero()
            if count_non_zero == max_count_non_zero:
                break

            tc_matrix @= tc_matrix

        return tc_matrix

    def _get_next_states(self, states: Iterable[State], symbol: Symbol) -> Set[State]:
        """Get next states from states with given symbol

        Parameters
        ----------
        states : iterable of State
            The current states
        symbol : :class:`~pyformlang.finite_automaton.Symbol`
            The input symbol

        Returns
        -------
        states : Set[State]
            Reachable states
        """
        next_states = set()
        for state in states:
            for n_state in self._transition_function(state, symbol):
                next_states.add(n_state)

        return next_states

    def _transition_function(self, state: State, symbol: Symbol) -> Set[State]:
        """Get states reachable from a state with given symbol

        Parameters
        ----------
        state : :class:`~pyformlang.finite_automaton.State`
            The source state
        symbol : :class:`~pyformlang.finite_automaton.Symbol`
            The input symbol

        Returns
        -------
        states : Set[State]
            Reachable states
        """
        next_states = set()
        for n_state in self.states:
            if self.is_transition(state, symbol, n_state):
                next_states.add(n_state)

        return next_states

    def __str__(self) -> str:
        lines = [
            "Adjacency Matrix FA: (",
            f"  States: {self._states_ids}",
            f"  Symbols: {self._symbols}",
            f"  Start states: {self._start_states}",
            f"  Final states: {self._final_states}",
            "  Adjacency matrices: (",
        ]

        for sym, matrix in self._adjacency_matrices.items():
            lines.append(f'    "{sym}":')

            matrix_array = matrix.toarray()
            for i, row in enumerate(matrix_array):
                lines.append(f"      {i}: {row}")

        lines.append("  )")
        lines.append(")")
        return "\n".join(lines)


def intersect_automata(
    automaton1: AdjacencyMatrixFA,
    automaton2: AdjacencyMatrixFA,
) -> AdjacencyMatrixFA:
    """Compute the intersection of two finite automata using tensor product

    Parameters
    ----------
    automaton1 : AdjacencyMatrixFA
        First  finite automaton
    automaton2 : AdjacencyMatrixFA
        Second finite automaton

    Returns
    -------
    automation : AdjacencyMatrixFA
        A new automaton representing the intersection
    """
    states_ids, adjacency_matrices = dict(), dict()
    start_states, final_states = set(), set()
    symbols = automaton1.symbols.intersection(automaton2.symbols)

    for symbol in symbols:
        adjacency_matrices[symbol] = kron(
            automaton1.adjacency_matrices[symbol],
            automaton2.adjacency_matrices[symbol],
            format="csr",
        ).astype(bool)

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
        adjacency_matrices=adjacency_matrices,
        start_states=start_states,
        final_states=final_states,
    )


def tensor_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: Set[int], final_nodes: Set[int]
) -> Set[Tuple[int, int]]:
    """Evaluate regular path queries using tensor product

    Parameters
    ----------
    regex : str
        Regular expression defining the path constraint
    graph : :class:`~networkx.MultiDiGraph`
        Graph where edges are labeled with symbols
    start_nodes : Set[int]
        Set of start nodes
    final_nodes : Set[int]
        Set of final nodes

    Returns
    -------
    pairs : Set[Tuple[int, int]]
        Set of node pairs (start, final) connected by paths matching the regex
    """
    nfa = graph_to_nfa(graph, start_nodes, final_nodes)
    amfa_graph = AdjacencyMatrixFA(nfa)

    dfa = regex_to_dfa(regex)
    amfa_request = AdjacencyMatrixFA(dfa)
    m = amfa_request.count_states

    amfa_front = intersect_automata(amfa_graph, amfa_request)
    tc_front = amfa_front.transition_closure()

    pairs = set()
    for s_st_graph, s_id_graph in amfa_graph.start_states_ids.items():
        for f_st_graph, f_id_graph in amfa_graph.final_states_ids.items():
            for _, s_id_request in amfa_request.start_states_ids.items():
                for _, f_id_request in amfa_request.final_states_ids.items():
                    if tc_front[
                        s_id_graph * m + s_id_request,
                        f_id_graph * m + f_id_request,
                    ]:
                        pairs.add((s_st_graph.value, f_st_graph.value))

    return pairs


def ms_bfs(
    automaton1: AdjacencyMatrixFA,
    automaton2: AdjacencyMatrixFA,
) -> Set[Tuple[Tuple[Any, Any], Tuple[int, int]]]:
    """Compute BFS of two finite automata using the multiple sources algorithm

    Parameters
    ----------
    automaton1 : AdjacencyMatrixFA
        First  finite automaton
    automaton2 : AdjacencyMatrixFA
        Second finite automaton

    Returns
    -------
    pairs : Set[Tuple[Tuple[Any, Any], Tuple[int, int]]]
        Set of pairs (start_value, final_value) and (start_id, final_id) connected by paths
    """
    n = automaton1.count_states
    m = automaton2.count_states

    fronts, matrices_visited = [], []
    for _, s_id1 in automaton1.start_states_ids.items():
        front = lil_array((n, m), dtype=bool)
        for _, s_id2 in automaton2.start_states_ids.items():
            front[s_id1, s_id2] = True

        fronts.append(front)
        matrices_visited.append(front.copy())

    symbols = automaton1.symbols.intersection(automaton2.symbols)
    adj_matrices1_tr = dict(
        (sym, m.transpose()) for sym, m in automaton1.adjacency_matrices.items()
    )
    matrix_true = lil_array((n, m), dtype=bool)
    matrix_true[:, :] = True

    while any(front.count_nonzero() for front in fronts):
        for i in range(len(fronts)):
            if not fronts[i].count_nonzero():
                continue

            for symbol in symbols:
                adj_matrix1_tr = adj_matrices1_tr[symbol]
                adj_matrix2 = automaton2.adjacency_matrices[symbol]
                fronts[i] += adj_matrix1_tr @ fronts[i] @ adj_matrix2

            fronts[i] = matrix_true - ((matrix_true - fronts[i]) + matrices_visited[i])
            matrices_visited[i] += fronts[i]

    final_states2 = lil_array((m, 1), dtype=bool)
    for _, f_id in automaton2.final_states_ids.items():
        final_states2[f_id, 0] = True

    pairs = set()
    for i, (s_state, s_id) in enumerate(automaton1.start_states_ids.items()):
        final_states1 = matrices_visited[i] @ final_states2

        for f_state, f_id in automaton1.final_states_ids.items():
            if final_states1[f_id, 0]:
                pairs.add(((s_state.value, f_state.value), (s_id, f_id)))

    return pairs


def ms_bfs_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: Set[int], final_nodes: Set[int]
) -> Set[Tuple[int, int]]:
    """Evaluate regular path queries using multi-source BFS

    Parameters
    ----------
    regex : str
        Regular expression defining the path constraint
    graph : :class:`~networkx.MultiDiGraph`
        Graph where edges are labeled with symbols
    start_nodes : Set[int]
        Set of start nodes
    final_nodes : Set[int]
        Set of final nodes

    Returns
    -------
    pairs : Set[Tuple[int, int]]
        Set of node pairs (start, final) connected by paths matching the regex
    """
    nfa = graph_to_nfa(graph, start_nodes, final_nodes)
    amfa_graph = AdjacencyMatrixFA(nfa)

    dfa = regex_to_dfa(regex)
    amfa_request = AdjacencyMatrixFA(dfa)

    pairs = ms_bfs(amfa_graph, amfa_request)
    return {values for values, _ in pairs}
