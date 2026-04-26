"""
Smoke tests for the GraphCode pipeline.
These run without any external dependencies (tree-sitter, sentence-transformers, etc.)
by testing the structural core against synthetic Python source files.
"""
from __future__ import annotations

import textwrap
import tempfile
import os
from pathlib import Path


def _write_tmp(files: dict[str, str]) -> str:
    """Write a dict of {rel_path: content} to a temp directory, return root."""
    tmp = tempfile.mkdtemp()
    for rel, content in files.items():
        full = Path(tmp) / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(textwrap.dedent(content))
    return tmp


# ---------------------------------------------------------------------------
# Phase 1: structure
# ---------------------------------------------------------------------------

def test_walk_tree_finds_python_files():
    from graphcode.phases.structure import walk_tree
    root = _write_tmp({
        "main.py": "x = 1",
        "src/utils.py": "y = 2",
        "node_modules/ignored.js": "z = 3",
    })
    sm = walk_tree(root)
    rel_paths = {f.rel_path for f in sm.files}
    assert "main.py" in rel_paths
    assert "src/utils.py" in rel_paths
    # node_modules should be pruned
    assert not any("node_modules" in p for p in rel_paths)


# ---------------------------------------------------------------------------
# Phase 2: AST parsing
# ---------------------------------------------------------------------------

def test_parse_python_functions():
    try:
        from graphcode.phases.ast_parser import parse_file
    except ImportError:
        return  # tree-sitter not installed in CI — skip

    root = _write_tmp({"module.py": """
        def greet(name):
            return f"Hello {name}"

        class Greeter:
            def hello(self):
                return greet("world")
    """})
    pf = parse_file(str(Path(root) / "module.py"))
    assert pf is not None
    names = {s.name for s in pf.symbols}
    assert "greet" in names
    assert "Greeter" in names
    assert "Greeter.hello" in names


def test_parse_python_imports():
    try:
        from graphcode.phases.ast_parser import parse_file
    except ImportError:
        return

    root = _write_tmp({"mod.py": """
        import os
        from pathlib import Path
        from . import utils
    """})
    pf = parse_file(str(Path(root) / "mod.py"))
    assert pf is not None
    modules = {imp["module"] for imp in pf.imports}
    assert "os" in modules
    assert "pathlib" in modules


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def test_graph_build_nodes_and_edges():
    try:
        from graphcode.phases.ast_parser import parse_file
        from graphcode.phases.graph_builder import build_graph
        from graphcode.models import NodeType, EdgeType
    except ImportError:
        return

    root = _write_tmp({"app.py": """
        def start():
            run()

        def run():
            pass
    """})
    pf = parse_file(str(Path(root) / "app.py"))
    assert pf is not None
    graph = build_graph([pf])

    # FILE node exists
    assert graph.has_node(str(Path(root) / "app.py"))
    # Function nodes exist
    file_nodes = graph.nodes(NodeType.FILE)
    func_nodes = graph.nodes(NodeType.FUNCTION)
    assert len(file_nodes) >= 1
    assert len(func_nodes) >= 2


# ---------------------------------------------------------------------------
# Traversal
# ---------------------------------------------------------------------------

def test_bfs_dfs_basics():
    from graphcode.graph.code_graph import CodeGraph
    from graphcode.graph.traversal import bfs, dfs
    from graphcode.models import NodeType, EdgeType, SymbolNode, SymbolEdge

    g = CodeGraph()
    for name in ["a", "b", "c"]:
        g.add_node(SymbolNode(id=name, name=name, node_type=NodeType.FUNCTION, file_path="f.py"))
    g.add_edge(SymbolEdge(source_id="a", target_id="b", edge_type=EdgeType.CALLS))
    g.add_edge(SymbolEdge(source_id="b", target_id="c", edge_type=EdgeType.CALLS))

    bfs_result = [n for n, _ in bfs(g, "a")]
    assert bfs_result == ["a", "b", "c"]

    dfs_result = [n for n, _ in dfs(g, "a")]
    assert dfs_result == ["a", "b", "c"]


def test_shortest_path():
    from graphcode.graph.code_graph import CodeGraph
    from graphcode.graph.traversal import shortest_path
    from graphcode.models import NodeType, EdgeType, SymbolNode, SymbolEdge

    g = CodeGraph()
    for name in ["x", "y", "z"]:
        g.add_node(SymbolNode(id=name, name=name, node_type=NodeType.FUNCTION, file_path="f.py"))
    g.add_edge(SymbolEdge(source_id="x", target_id="y", edge_type=EdgeType.CALLS))
    g.add_edge(SymbolEdge(source_id="y", target_id="z", edge_type=EdgeType.CALLS))

    path = shortest_path(g, "x", "z")
    assert path == ["x", "y", "z"]

    assert shortest_path(g, "z", "x") is None  # no back-edge


def test_coverage_tour():
    from graphcode.graph.code_graph import CodeGraph
    from graphcode.graph.traversal import coverage_tour
    from graphcode.models import NodeType, EdgeType, SymbolNode, SymbolEdge

    g = CodeGraph()
    nodes = ["n1", "n2", "n3"]
    for n in nodes:
        g.add_node(SymbolNode(id=n, name=n, node_type=NodeType.FUNCTION, file_path="f.py"))
    g.add_edge(SymbolEdge(source_id="n1", target_id="n2", edge_type=EdgeType.CALLS))
    g.add_edge(SymbolEdge(source_id="n2", target_id="n3", edge_type=EdgeType.CALLS))

    tour = coverage_tour(g, nodes)
    assert set(tour) == set(nodes)
    assert len(tour) == 3


# ---------------------------------------------------------------------------
# Clustering fallback
# ---------------------------------------------------------------------------

def test_cluster_fallback():
    from graphcode.phases.ast_parser import ParsedFile
    from graphcode.phases.graph_builder import build_graph
    from graphcode.phases.clustering import cluster_graph
    from graphcode.models import NodeType, SymbolNode

    # Build a minimal graph without needing tree-sitter
    from graphcode.graph.code_graph import CodeGraph
    from graphcode.models import EdgeType, SymbolEdge

    g = CodeGraph()
    for i in range(5):
        g.add_node(SymbolNode(id=f"sym{i}", name=f"sym{i}", node_type=NodeType.FUNCTION, file_path=f"f{i%2}.py"))
    g.add_edge(SymbolEdge(source_id="sym0", target_id="sym1", edge_type=EdgeType.CALLS))
    g.add_edge(SymbolEdge(source_id="sym2", target_id="sym3", edge_type=EdgeType.CALLS))

    clusters = cluster_graph(g)
    assert len(clusters) >= 1
    total_nodes = sum(len(c.node_ids) for c in clusters)
    assert total_nodes == 5
