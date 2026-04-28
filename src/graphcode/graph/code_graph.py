"""
CodeGraph: directed multigraph of source symbols and their relationships.

Backed by NetworkX MultiDiGraph. All graph algorithms operate through this
wrapper so the underlying library can be swapped without touching callers.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx

from graphcode.models import EdgeType, NodeType, SymbolEdge, SymbolNode


class CodeGraph:
    def __init__(self) -> None:
        self._g: nx.MultiDiGraph = nx.MultiDiGraph()

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_node(self, node: SymbolNode) -> None:
        self._g.add_node(node.id, **{
            "name": node.name,
            "node_type": node.node_type.value,
            "file_path": node.file_path,
            "line_start": node.line_start,
            "line_end": node.line_end,
            "signature": node.signature,
            "docstring": node.docstring,
            "language": node.language,
            **node.metadata,
        })

    def has_node(self, node_id: str) -> bool:
        return self._g.has_node(node_id)

    def get_node_data(self, node_id: str) -> dict[str, Any] | None:
        if not self._g.has_node(node_id):
            return None
        return dict(self._g.nodes[node_id])

    def nodes(self, node_type: NodeType | None = None) -> list[str]:
        if node_type is None:
            return list(self._g.nodes())
        return [
            n for n, d in self._g.nodes(data=True)
            if d.get("node_type") == node_type.value
        ]

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_edge(self, edge: SymbolEdge) -> None:
        if not (self._g.has_node(edge.source_id) and self._g.has_node(edge.target_id)):
            return
        if self.has_edge(edge.source_id, edge.target_id, edge.edge_type):
            return
        self._g.add_edge(
            edge.source_id,
            edge.target_id,
            edge_type=edge.edge_type.value,
            line=edge.line,
            **edge.metadata,
        )

    def has_edge(self, source_id: str, target_id: str, edge_type: EdgeType | None = None) -> bool:
        if not self._g.has_edge(source_id, target_id):
            return False
        if edge_type is None:
            return True
        for d in self._g[source_id][target_id].values():
            if d.get("edge_type") == edge_type.value:
                return True
        return False

    def remove_edge(self, source_id: str, target_id: str, edge_type: EdgeType | None = None) -> None:
        """Remove edges between source and target. If edge_type given, only remove that type."""
        if not self._g.has_edge(source_id, target_id):
            return
        if edge_type is None:
            self._g.remove_edges_from([(source_id, target_id)])
            return
        keys_to_remove = [
            k for k, d in self._g[source_id][target_id].items()
            if d.get("edge_type") == edge_type.value
        ]
        for k in keys_to_remove:
            self._g.remove_edge(source_id, target_id, key=k)

    def edges(
        self,
        source: str | None = None,
        edge_type: EdgeType | None = None,
    ) -> list[tuple[str, str, str]]:
        raw = (
            self._g.out_edges(source, data=True)
            if source and self._g.has_node(source)
            else self._g.edges(data=True)
        )
        result: list[tuple[str, str, str]] = []
        for u, v, d in raw:
            et = d.get("edge_type", "")
            if edge_type is None or et == edge_type.value:
                result.append((u, v, et))
        return result

    # ------------------------------------------------------------------
    # Degree & adjacency
    # ------------------------------------------------------------------

    def in_degree(self, node_id: str) -> int:
        return self._g.in_degree(node_id) if self._g.has_node(node_id) else 0

    def out_degree(self, node_id: str) -> int:
        return self._g.out_degree(node_id) if self._g.has_node(node_id) else 0

    def successors(self, node_id: str) -> list[str]:
        if not self._g.has_node(node_id):
            return []
        return list(self._g.successors(node_id))

    def predecessors(self, node_id: str) -> list[str]:
        if not self._g.has_node(node_id):
            return []
        return list(self._g.predecessors(node_id))

    # ------------------------------------------------------------------
    # Graph-level properties
    # ------------------------------------------------------------------

    def node_count(self) -> int:
        return self._g.number_of_nodes()

    def edge_count(self) -> int:
        return self._g.number_of_edges()

    @property
    def nx_graph(self) -> nx.MultiDiGraph:
        return self._g

    def to_simple_digraph(self) -> nx.DiGraph:
        """Collapse multi-edges to a single edge (needed by algorithms that require simple graphs)."""
        g: nx.DiGraph = nx.DiGraph()
        for n, d in self._g.nodes(data=True):
            g.add_node(n, **d)
        for u, v, d in self._g.edges(data=True):
            if not g.has_edge(u, v):
                g.add_edge(u, v, **d)
        return g

    def subgraph(self, node_ids: list[str]) -> "CodeGraph":
        sub = CodeGraph()
        sub._g = self._g.subgraph(node_ids).copy()
        return sub

    def stats(self) -> dict[str, Any]:
        node_types: dict[str, int] = {}
        for _, d in self._g.nodes(data=True):
            t = d.get("node_type", "unknown")
            node_types[t] = node_types.get(t, 0) + 1

        edge_types: dict[str, int] = {}
        for _, _, d in self._g.edges(data=True):
            t = d.get("edge_type", "unknown")
            edge_types[t] = edge_types.get(t, 0) + 1

        return {
            "nodes": self._g.number_of_nodes(),
            "edges": self._g.number_of_edges(),
            "node_types": node_types,
            "edge_types": edge_types,
        }

    def __repr__(self) -> str:
        s = self.stats()
        return f"<CodeGraph nodes={s['nodes']} edges={s['edges']}>"

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the graph to a JSON file."""
        data = {
            "nodes": [
                {"id": n, **dict(d)}
                for n, d in self._g.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v, **dict(d)}
                for u, v, d in self._g.edges(data=True)
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> "CodeGraph":
        """Load a graph from a JSON file."""
        raw = json.loads(Path(path).read_text())
        graph = cls()
        for node in raw["nodes"]:
            nid = node.pop("id")
            graph._g.add_node(nid, **node)
        for edge in raw["edges"]:
            src = edge.pop("source")
            tgt = edge.pop("target")
            graph._g.add_edge(src, tgt, **edge)
        return graph
