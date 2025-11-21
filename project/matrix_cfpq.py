from typing import Set, Tuple, Dict
from pyformlang.cfg import CFG, Terminal, Variable
from networkx import DiGraph

from scipy.sparse import lil_array

from project.hellings_cfpq import cfg_to_weak_normal_form


def matrix_based_cfpq(
    cfg: CFG,
    graph: DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> Set[Tuple[int, int]]:
    """Evaluate CFG path queries using the matrix algorithm

    Parameters
    ----------
    cfg : :class:`~pyformlang.cfg.CFG`
        Context-Free Grammar defining the path constraint
    graph : :class:`~networkx.DiGraph`
        Graph where edges are labeled with symbols
    start_nodes : Set[int]
        Set of start nodes
    final_nodes : Set[int]
        Set of final nodes

    Returns
    -------
    pairs : Set[Tuple[int, int]]
        Set of node pairs (start, final) connected by paths matching CFG
    """
    graph_id_to_v = []
    graph_v_to_id = {}
    for i, v in enumerate(graph.nodes):
        graph_id_to_v.append(v)
        graph_v_to_id[v] = i

    cwnf = cfg_to_weak_normal_form(cfg)

    n = graph.number_of_nodes()
    T: Dict[Variable, lil_array] = dict()
    xs: Dict[Terminal, Set[Variable]] = dict()
    ps: Dict[Variable, Set[Tuple[Variable, Variable]]] = dict()

    for p in cwnf.productions:
        A = p.head
        B = p.body

        if A not in T:
            T[A] = lil_array((n, n), dtype=bool)

        if len(B) == 1 and isinstance(x := B[0], Terminal):
            xs.setdefault(x, set()).add(A)
        elif len(B) == 2:
            ps.setdefault(A, set()).add(tuple(B))

    for v, u, t in graph.edges(data="label"):
        x = Terminal(t)
        As = xs.get(x, set())
        for A in As:
            i = graph_v_to_id[v]
            j = graph_v_to_id[u]
            T[A][i, j] = True

    for A in cwnf.get_nullable_symbols():
        T.setdefault(A, lil_array((n, n), dtype=bool)).setdiag(True)

    is_changed = True
    while is_changed:
        is_changed = False

        for A_i, As in ps.items():
            for A_j, A_k in As:
                cnz_old = T[A_i].count_nonzero()
                T[A_i] += T[A_j] @ T[A_k]

                if cnz_old != T[A_i].count_nonzero():
                    is_changed = True

    if start_nodes is None:
        start_nodes = set(graph.nodes)
    if final_nodes is None:
        final_nodes = set(graph.nodes)

    pairs = set()
    matrix = T[cwnf.start_symbol]

    rows, cols = matrix.nonzero()
    for i, j in zip(rows, cols):
        s, f = graph_id_to_v[i], graph_id_to_v[j]
        if s in start_nodes and f in final_nodes:
            pairs.add((s, f))

    return pairs
