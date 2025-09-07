from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cfpq_data as cd
import networkx as nx


@dataclass
class MetadataGraph:
    count_nodes: int
    count_edges: int
    tags_edges: set[any]


def get_metadata(name_graph: str) -> MetadataGraph:
    """Loads graph metadata from dataset.

    Parameters
    ----------
    name_graph : str
        The name of the graph from the dataset.

    Returns
    -------
    metadata : MetadataGraph
        Graph metadata.
    """
    archive = cd.download(name_graph)
    graph = cd.graph_from_csv(archive)

    return MetadataGraph(
        graph.number_of_nodes(),
        graph.number_of_edges(),
        set(cd.get_sorted_labels(graph)),
    )


def build_graph_two_cycles(n: int, m: int, labels: Tuple[str, str], path_save: Path):
    """Builds a graph with two cycles connected by one node. With labeled edges.
    Saves it into DOT file with given path.

    Parameters
    ----------
    n : int
        The number of nodes in the first cycle without a common node.

    m : int
        The number of nodes in the second cycle without a common node.

    labels: Tuple[str, str]
        Labels that will be used to mark the edges of the graph.

    path_save : str or file
        Filename or file handle for saving.
    """
    graph = cd.labeled_two_cycles_graph(n, m, labels=labels)
    graph_pydot = nx.drawing.nx_pydot.to_pydot(graph)
    graph_pydot.write(path_save, prog="dot", format="dot")
