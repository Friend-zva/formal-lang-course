from pyformlang.cfg import CFG, Variable, Terminal, Production, Epsilon
from networkx import DiGraph

from project.matrix_cfpq import matrix_based_cfpq


def test_matrix_based_cfpq():
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
    var_a = Variable("A")
    var_b = Variable("B")
    t_a = Terminal("a")
    t_b = Terminal("b")
    cfg = CFG(
        variables={var_s, var_a, var_b},
        terminals={t_a, t_b},
        productions={
            Production(var_s, [t_a]),
            Production(var_s, [Epsilon()]),
            Production(var_s, [t_b, t_a, var_s]),
        },
        start_symbol=var_s,
    )

    assert matrix_based_cfpq(cfg, graph, {0}, {0}) == {(0, 0)}
