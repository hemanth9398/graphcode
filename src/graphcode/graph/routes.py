"""
Route extraction: trace execution territories from entry points via DFS.

A Route is the complete set of symbols reachable from one entry point —
like a GPS route map that answers "if I call X, what does the codebase touch?"
"""
from __future__ import annotations

from graphcode.graph.code_graph import CodeGraph
from graphcode.graph.traversal import dfs, find_entry_points
from graphcode.models import Route


def extract_routes(
    graph: CodeGraph,
    entry_ids: list[str] | None = None,
    max_depth: int = 15,
) -> list[Route]:
    """
    Trace DFS routes from every entry point (or a supplied list).

    Each Route captures:
    - The ordered DFS traversal of reachable nodes
    - All edges whose both endpoints are within that territory
    - The maximum traversal depth
    """
    entries = entry_ids if entry_ids is not None else find_entry_points(graph)
    routes: list[Route] = []

    for entry_id in entries:
        if not graph.has_node(entry_id):
            continue

        dfs_result = dfs(graph, entry_id, max_depth=max_depth)
        if not dfs_result:
            continue

        node_ids = [n for n, _ in dfs_result]
        territory = set(node_ids)

        # Collect only the edges that stay inside this route's territory
        internal_edges: list[tuple[str, str, str]] = [
            (u, v, et)
            for u, v, et in graph.edges()
            if u in territory and v in territory
        ]

        max_d = max(d for _, d in dfs_result)
        entry_data = graph.get_node_data(entry_id) or {}

        routes.append(Route(
            entry_id=entry_id,
            entry_name=entry_data.get("name", entry_id),
            nodes=node_ids,
            edges=internal_edges,
            depth=max_d,
        ))

    # Sort: largest territory first (most impactful entry points)
    routes.sort(key=lambda r: len(r.nodes), reverse=True)
    return routes


def build_route_map(routes: list[Route]) -> dict[str, Route]:
    """Index routes by entry_id for O(1) lookup."""
    return {r.entry_id: r for r in routes}


def route_diff(r1: Route, r2: Route) -> dict[str, list[str]]:
    """What symbols are unique to each route vs. shared."""
    s1, s2 = set(r1.nodes), set(r2.nodes)
    return {
        "only_in_r1": sorted(s1 - s2),
        "only_in_r2": sorted(s2 - s1),
        "shared": sorted(s1 & s2),
    }


def route_overlap_matrix(routes: list[Route]) -> list[list[float]]:
    """
    Jaccard overlap between every pair of routes.
    Returns an N×N matrix where entry [i][j] is the Jaccard index.
    """
    n = len(routes)
    sets = [set(r.nodes) for r in routes]
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 1.0
                continue
            inter = len(sets[i] & sets[j])
            union = len(sets[i] | sets[j])
            matrix[i][j] = inter / union if union else 0.0
    return matrix
