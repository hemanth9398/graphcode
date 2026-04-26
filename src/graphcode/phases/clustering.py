"""
Phase 5: Leiden community detection on the code graph.

Leiden produces high-quality, well-separated communities.
Falls back to file-based grouping if leidenalg/igraph are not installed.
"""
from __future__ import annotations

from graphcode.graph.code_graph import CodeGraph
from graphcode.models import Cluster


def _leiden_partition(graph: CodeGraph) -> list[list[str]]:
    try:
        import igraph as ig
        import leidenalg

        nodes = list(graph.nodes())
        if not nodes:
            return []

        idx = {n: i for i, n in enumerate(nodes)}
        # Build undirected igraph from the directed CodeGraph
        edge_pairs = [
            (idx[u], idx[v])
            for u, v, _ in graph.edges()
            if u in idx and v in idx
        ]
        if not edge_pairs:
            return [[n] for n in nodes]

        g_ig = ig.Graph(n=len(nodes), edges=edge_pairs, directed=False)
        partition = leidenalg.find_partition(
            g_ig, leidenalg.ModularityVertexPartition, seed=42
        )
        return [[nodes[i] for i in community] for community in partition]

    except (ImportError, Exception):
        return _file_clusters(graph)


def _file_clusters(graph: CodeGraph) -> list[list[str]]:
    """Fallback: group symbols by source file."""
    groups: dict[str, list[str]] = {}
    for node_id in graph.nodes():
        data = graph.get_node_data(node_id) or {}
        key = data.get("file_path") or "__external__"
        groups.setdefault(key, []).append(node_id)
    return list(groups.values())


def _cohesion(graph: CodeGraph, node_ids: list[str]) -> float:
    """Internal-edge ratio: edges within cluster / max possible edges."""
    n = len(node_ids)
    if n < 2:
        return 1.0
    inside = set(node_ids)
    internal = sum(1 for u, v, _ in graph.edges() if u in inside and v in inside)
    return internal / (n * (n - 1))


def _dominant_label(graph: CodeGraph, node_ids: list[str]) -> str:
    counts: dict[str, int] = {}
    for nid in node_ids:
        fp = (graph.get_node_data(nid) or {}).get("file_path", "")
        if fp:
            counts[fp] = counts.get(fp, 0) + 1
    if not counts:
        return ""
    dominant = max(counts, key=lambda k: counts[k])
    return dominant.rsplit("/", 1)[-1].rsplit(".", 1)[0]  # bare filename without ext


def cluster_graph(graph: CodeGraph) -> list[Cluster]:
    """Run Leiden (or fallback) and return annotated Cluster objects."""
    raw = _leiden_partition(graph)
    clusters: list[Cluster] = []

    for i, node_ids in enumerate(raw):
        if not node_ids:
            continue
        clusters.append(Cluster(
            id=i,
            node_ids=node_ids,
            cohesion_score=_cohesion(graph, node_ids),
            label=_dominant_label(graph, node_ids),
        ))

    # Largest clusters first
    clusters.sort(key=lambda c: len(c.node_ids), reverse=True)
    # Re-assign sequential IDs after sort
    for new_id, c in enumerate(clusters):
        c.id = new_id

    return clusters
