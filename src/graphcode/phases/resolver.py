"""
Phase 4: Cross-file resolution.

Upgrades FILE --imports--> MODULE(placeholder) edges to
FILE --imports--> FILE edges wherever the import can be traced to an actual
source file already present in the graph.
"""
from __future__ import annotations

from pathlib import Path

from graphcode.graph.code_graph import CodeGraph
from graphcode.models import EdgeType, NodeType, SymbolEdge

_SOURCE_EXTS = [".py", ".ts", ".tsx", ".js", ".jsx", ".mjs"]


def _resolve_module(module: str, importer: str, root: str) -> str | None:
    """
    Try to resolve *module* (as written in source) to an absolute file path.

    Strategy:
      1. Relative imports ("./foo", "../bar") — resolve from importer directory.
      2. Absolute bare imports — try root/<module> with source extensions.
      3. Package-style ("src/utils") — same as (2) but with slash mapping.
    """
    importer_dir = Path(importer).parent

    if module.startswith("."):
        # Relative
        base = (importer_dir / module).resolve()
        for ext in [""] + _SOURCE_EXTS:
            p = Path(str(base) + ext)
            if p.exists():
                return str(p)
        # Maybe it's a directory with an index file
        for ext in _SOURCE_EXTS:
            p = base / f"index{ext}"
            if p.exists():
                return str(p)
    else:
        # Bare / package import — look under root
        parts = module.replace(".", "/").replace("-", "_")
        for base in [Path(root), Path(root) / "src"]:
            candidate = base / parts
            for ext in [""] + _SOURCE_EXTS:
                p = Path(str(candidate) + ext)
                if p.exists():
                    return str(p)
            for ext in _SOURCE_EXTS:
                p = candidate / f"index{ext}"
                if p.exists():
                    return str(p)

    return None


def resolve_imports(graph: CodeGraph, root: str) -> CodeGraph:
    """
    Walk all unresolved import edges and, where possible, replace the
    MODULE placeholder target with the real FILE node.
    """
    new_edges: list[SymbolEdge] = []
    stale_edges: list[tuple[str, str]] = []

    for src, dst, _ in graph.edges(edge_type=EdgeType.IMPORTS):
        src_data = graph.get_node_data(src) or {}
        dst_data = graph.get_node_data(dst) or {}

        # Only upgrade FILE --> MODULE(placeholder) edges
        if src_data.get("node_type") != NodeType.FILE.value:
            continue
        if dst_data.get("node_type") != NodeType.MODULE.value:
            continue

        module_name = dst_data.get("name", "")
        resolved = _resolve_module(module_name, src, root)

        if resolved and graph.has_node(resolved):
            if not graph.has_edge(src, resolved, EdgeType.IMPORTS):
                new_edges.append(SymbolEdge(
                    source_id=src,
                    target_id=resolved,
                    edge_type=EdgeType.IMPORTS,
                ))
            stale_edges.append((src, dst))

    for edge in new_edges:
        graph.add_edge(edge)

    for src, dst in stale_edges:
        graph.remove_edge(src, dst, EdgeType.IMPORTS)

    return graph
