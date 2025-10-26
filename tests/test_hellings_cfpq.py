from pyformlang.cfg import CFG, Variable, Terminal, Production, Epsilon
from networkx import DiGraph

from project.hellings_cfpq import cfg_to_weak_normal_form, hellings_based_cfpq


def test_cfg_to_weak_normal_form():
    var_s = Variable("S")
    t_a = Terminal("a")
    t_b = Terminal("b")
    cfg = CFG(
        variables={var_s},
        terminals={t_a, t_b},
        productions={
            Production(var_s, [Epsilon()]),
            Production(var_s, [t_a]),
            Production(var_s, [t_b, t_a, var_s]),
        },
        start_symbol=var_s,
    )

    cwnf = cfg_to_weak_normal_form(cfg)

    var_s1 = Variable("C#CNF#1")
    var_a = Variable("a#CNF#")
    var_b = Variable("b#CNF#")

    assert set(cwnf.variables) == {var_s, var_s1, var_a, var_b}
    assert set(cwnf.terminals) == {t_a, t_b}
    assert set(cwnf.productions) == {
        Production(var_s, [Epsilon()]),
        Production(var_s, [t_a]),
        Production(var_s, [var_b, var_s1]),
        Production(var_s, [var_b, var_a]),
        Production(var_a, [t_a]),
        Production(var_b, [t_b]),
        Production(var_s1, [var_a, var_s]),
    }
    assert cwnf.start_symbol == var_s


def test_hellings_based_cfpq():
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

    assert hellings_based_cfpq(cfg, graph, {0}, {0}) == {(0, 0)}
