"""
Graph traversal algorithms on a CodeGraph.

  BFS  — breadth-first discovery of all reachable symbols
  DFS  — depth-first pre-order traversal (mirrors execution order)
  shortest_path / all_paths — classic pathfinding
  find_entry_points — locate natural execution roots
  coverage_tour  — greedy nearest-neighbour TSP to cover a cluster
"""
from __future__ import annotations

from collections import deque
from typing import Any

import networkx as nx

from graphcode.graph.code_graph import CodeGraph
from graphcode.models import EdgeType


def bfs(
    graph: CodeGraph,
    start: str,
    max_depth: int = 10,
    edge_filter: EdgeType | None = None,
) -> list[tuple[str, int]]:
    """Return (node_id, depth) pairs in BFS order from *start*."""
    if not graph.has_node(start):
        return []

    visited: dict[str, int] = {start: 0}
    queue: deque[tuple[str, int]] = deque([(start, 0)])
    result: list[tuple[str, int]] = [(start, 0)]

    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for _, nbr, d in graph.nx_graph.out_edges(node, data=True):
            if edge_filter and d.get("edge_type") != edge_filter.value:
                continue
            if nbr not in visited:
                visited[nbr] = depth + 1
                queue.append((nbr, depth + 1))
                result.append((nbr, depth + 1))

    return result


def dfs(
    graph: CodeGraph,
    start: str,
    max_depth: int = 20,
    edge_filter: EdgeType | None = None,
) -> list[tuple[str, int]]:
    """Return (node_id, depth) pairs in DFS pre-order from *start*."""
    if not graph.has_node(start):
        return []

    visited: set[str] = set()
    result: list[tuple[str, int]] = []

    def _recurse(node: str, depth: int) -> None:
        if node in visited or depth > max_depth:
            return
        visited.add(node)
        result.append((node, depth))
        for _, nbr, d in graph.nx_graph.out_edges(node, data=True):
            if edge_filter and d.get("edge_type") != edge_filter.value:
                continue
            _recurse(nbr, depth + 1)

    _recurse(start, 0)
    return result


def shortest_path(graph: CodeGraph, source: str, target: str) -> list[str] | None:
    """Shortest hop-count path between two symbols; None if unreachable."""
    try:
        return nx.shortest_path(graph.nx_graph, source, target)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def all_paths(
    graph: CodeGraph,
    source: str,
    target: str,
    max_paths: int = 10,
    cutoff: int = 8,
) -> list[list[str]]:
    """All simple paths from *source* to *target* up to *cutoff* hops."""
    try:
        gen = nx.all_simple_paths(graph.nx_graph, source, target, cutoff=cutoff)
        return [p for _, p in zip(range(max_paths), gen)]
    except (nx.NetworkXError, nx.NodeNotFound):
        return []


def reachable_from(
    graph: CodeGraph,
    start: str,
    edge_types: list[EdgeType] | None = None,
) -> set[str]:
    """Set of all node IDs reachable from *start* via specified edge types."""
    et_values = {e.value for e in edge_types} if edge_types else None
    visited: set[str] = set()
    queue: deque[str] = deque([start])

    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        for _, nbr, d in graph.nx_graph.out_edges(node, data=True):
            if et_values is None or d.get("edge_type") in et_values:
                if nbr not in visited:
                    queue.append(nbr)

    return visited


def find_cycles(graph: CodeGraph) -> list[list[str]]:
    """Return all simple cycles (circular dependencies) in the graph."""
    return list(nx.simple_cycles(graph.to_simple_digraph()))


def topological_sort(graph: CodeGraph) -> list[str] | None:
    """Topological ordering of nodes; None if the graph contains cycles."""
    try:
        return list(nx.topological_sort(graph.to_simple_digraph()))
    except nx.NetworkXUnfeasible:
        return None


def find_entry_points(graph: CodeGraph) -> list[str]:
    """
    Natural execution roots: nodes with no in-edges, plus known entry names.
    Entry names: main, index, app, server, routes, run, start, handler, init.
    """
    _ENTRY_NAMES = {"main", "index", "app", "server", "routes", "run", "start",
                    "handler", "init", "bootstrap", "setup", "entrypoint"}
    seen: set[str] = set()
    result: list[str] = []

    for node_id in graph.nodes():
        data = graph.get_node_data(node_id) or {}
        name_lower = data.get("name", "").lower().split(".")[-1]  # strip class prefix

        is_entry_name = name_lower in _ENTRY_NAMES
        is_root = graph.in_degree(node_id) == 0

        if (is_root or is_entry_name) and node_id not in seen:
            seen.add(node_id)
            result.append(node_id)

    return result


def coverage_tour(graph: CodeGraph, nodes: list[str]) -> list[str]:
    """
    Greedy nearest-neighbour tour through *nodes* (TSP approximation).

    Visits every node in the set by always moving to the nearest unvisited
    neighbour (direct successor first, then by shortest graph distance).
    Useful for generating a minimal "reading order" for a cluster.
    """
    if not nodes:
        return []

    unvisited = set(nodes)
    current = nodes[0]
    tour = [current]
    unvisited.discard(current)

    while unvisited:
        # Prefer direct successors already in the node set
        direct = set(graph.successors(current)) & unvisited
        if direct:
            next_node = next(iter(direct))
        else:
            # Fall back to closest by graph distance
            best: str | None = None
            best_dist = float("inf")
            for candidate in unvisited:
                path = shortest_path(graph, current, candidate)
                if path and len(path) < best_dist:
                    best_dist = len(path)
                    best = candidate
            next_node = best if best else next(iter(unvisited))

        tour.append(next_node)
        unvisited.discard(next_node)
        current = next_node

    return tour
