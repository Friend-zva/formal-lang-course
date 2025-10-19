from pyformlang.cfg import CFG, Production, Variable, Epsilon


def cfg_to_weak_normal_form(cfg: CFG) -> CFG:
    """Gets the Chomsky Weakened Normal Form of a Context Free Grammar

    Parameters
    ----------
    cfg : :class:`~pyformlang.cfg.CFG`
        An original CFG

    Returns
    -------
    cwnf : :class:`~pyformlang.cfg.CFG`
        A new CFG equivalent in the CWNF
    """
    cfg_nf = cfg.to_normal_form()

    prods_eps = set(cfg_nf.productions)
    null_syms = cfg.get_nullable_symbols()

    for var in null_syms:
        prods_eps.add(Production(Variable(var.value), Epsilon()))

    cwnf = CFG(
        variables=cfg_nf.variables,
        terminals=cfg_nf.terminals,
        start_symbol=cfg_nf.start_symbol,
        productions=prods_eps,
    )
    return cwnf.remove_useless_symbols()
