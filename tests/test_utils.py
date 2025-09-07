from project.utils import (
    get_metadata,
    build_graph_two_cycles,
)

import networkx as nx


def test_get_metadata():
    metadata = get_metadata("generations")
    assert metadata.count_nodes == 129
    assert metadata.count_edges == 273
    assert metadata.tags_edges == {
        "rest",
        "type",
        "first",
        "onProperty",
        "intersectionOf",
        "equivalentClass",
        "hasValue",
        "hasChild",
        "hasParent",
        "inverseOf",
        "sameAs",
        "hasSibling",
        "oneOf",
        "someValuesFrom",
        "hasSex",
        "range",
        "versionInfo",
    }


def test_build_graph_two_cycles():
    path = "tests/graphs/result/two_cycles.dot"
    build_graph_two_cycles(4, 6, ("a", "b"), path)
    assert nx.utils.graphs_equal(
        nx.nx_pydot.read_dot(path),
        nx.nx_pydot.read_dot("tests/graphs/source/two_cycles.dot"),
    )
