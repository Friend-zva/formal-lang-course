from typing import Set, Tuple
from pyformlang.cfg import CFG, Production, Epsilon, Terminal
from networkx import DiGraph

from collections import deque


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


def hellings_based_cfpq(
    cfg: CFG,
    graph: DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> Set[Tuple[int, int]]:
    """Evaluate CFG path queries using the Hellings algorithm

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
    cwnf = cfg_to_weak_normal_form(cfg)

    result = set()
    process = deque()

    for var in cwnf.get_nullable_symbols():
        for n in graph.nodes:
            triple = (var, n, n)
            result.add(triple)
            process.append(triple)

    for v, u, t in graph.edges(data="label"):
        for p in cwnf.productions:
            if [Terminal(t)] != p.body:
                continue
            var = p.head

            triple = (var, v, u)
            result.add(triple)
            process.append(triple)

    while process:
        (var_i, v_, u_) = process.popleft()
        adder = set()

        for var_j, v1, v in result:
            if v != v_:
                continue

            for p in cwnf.productions:
                if [var_j, var_i] != p.body:
                    continue
                var_k = p.head

                triple = (var_k, v1, u_)
                if triple not in result:
                    process.append(triple)
                    adder.add(triple)

        for var_j, u, u1 in result:
            if u != u_:
                continue

            for p in cwnf.productions:
                if [var_i, var_j] != p.body:
                    continue
                var_k = p.head

                triple = (var_k, v_, u1)
                if triple not in result:
                    process.append(triple)
                    adder.add(triple)

        result = result.union(adder)

    if start_nodes is None:
        start_nodes = set(graph.nodes)
    if final_nodes is None:
        final_nodes = set(graph.nodes)

    pairs = set()
    for var, s, f in result:
        if var != cwnf.start_symbol:
            continue
        if s in start_nodes and f in final_nodes:
            pairs.add((s, f))

    return pairs
