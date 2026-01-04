from networkx import DiGraph

from pyformlang.cfg import CFG

from project.utils import cfg_to_rsm

from project.gll_cfpq import gll_based_cfpq
from project.tensor_cfpq import tensor_based_cfpq


def test_gll_based_cfpq():
    graph = DiGraph()
    graph.add_node(0, start_node=True)
    graph.add_edges_from(
        [
            (0, 1, {"label": "a"}),
            (1, 2, {"label": "a"}),
        ]
    )

    cfg = CFG.from_text("S -> a | S a | S S")
    rsm = cfg_to_rsm(cfg)

    assert gll_based_cfpq(rsm, graph, {0}, None) == {(0, 1), (0, 2)}


def test_cfpqs():
    graph = DiGraph()
    graph.add_node(0, start_node=True, final_node=True)
    graph.add_edges_from(
        [
            (0, 0, {"label": "a"}),
            (0, 0, {"label": "b"}),
        ]
    )

    cfg = CFG.from_text("S -> ε | a S | b S")
    rsm = cfg_to_rsm(cfg)

    assert gll_based_cfpq(rsm, graph, {0}, {0}) == tensor_based_cfpq(
        rsm, graph, {0}, {0}
    )
    assert gll_based_cfpq(rsm, graph, {0}, {0}) == {(0, 0)}
