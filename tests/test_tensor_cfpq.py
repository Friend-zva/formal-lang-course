from pyformlang.cfg import CFG, Variable, Terminal, Production, Epsilon
from networkx import MultiDiGraph, DiGraph


from project.tensor_cfpq import cfg_to_rsm, tensor_based_cfpq
from project.matrix_cfpq import matrix_based_cfpq


def test_tensor_based_cfpq():
    graph = DiGraph()
    graph.add_node(0, start_node=True, final_node=True)
    graph.add_edges_from(
        [
            (0, 0, {"label": "a"}),
            (0, 1, {"label": "b"}),
            (1, 0, {"label": "b"}),
            (1, 1, {"label": "a"}),
        ]
    )

    var_s = Variable("S")
    t_a = Terminal("a")
    t_b = Terminal("b")
    cfg = CFG(
        variables={var_s},
        terminals={t_a, t_b},
        productions={
            Production(var_s, [t_a, var_s, t_b]),
            Production(var_s, [Epsilon()]),
            Production(var_s, [t_a, t_b]),
        },
        start_symbol=var_s,
    )

    rsm = cfg_to_rsm(cfg)
    assert tensor_based_cfpq(rsm, graph, {0}, {0}) == {(0, 0)}


def test_from_article():
    graph = MultiDiGraph()
    graph.add_nodes_from([0, 1, 2, 3])
    graph.add_edges_from(
        [
            (0, 1, {"label": "a"}),
            (1, 2, {"label": "a"}),
            (2, 0, {"label": "a"}),
            (2, 3, {"label": "b"}),
            (3, 2, {"label": "b"}),
        ]
    )

    cfg = CFG.from_text("S -> a S b | a b")
    rsm = cfg_to_rsm(cfg)

    assert tensor_based_cfpq(rsm, graph) == {
        (1, 2),
        (0, 3),
        (2, 3),
        (0, 2),
        (2, 2),
        (1, 3),
    }


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

    assert tensor_based_cfpq(rsm, graph, {0}, {0}) == matrix_based_cfpq(
        cfg, graph, {0}, {0}
    )
    assert tensor_based_cfpq(rsm, graph, {0}, {0}) == {(0, 0)}
