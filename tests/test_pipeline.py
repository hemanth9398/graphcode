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

import pytest


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


def test_walk_tree_respects_extensions():
    from graphcode.phases.structure import walk_tree
    root = _write_tmp({
        "app.py": "x = 1",
        "readme.md": "# hi",
        "data.csv": "a,b,c",
    })
    sm = walk_tree(root)
    assert sm.file_count == 1
    assert sm.files[0].extension == ".py"


# ---------------------------------------------------------------------------
# Phase 2: AST parsing
# ---------------------------------------------------------------------------

def _require_tree_sitter():
    try:
        import tree_sitter
        import tree_sitter_python
    except ImportError:
        pytest.skip("tree-sitter not installed")


def test_parse_python_functions():
    _require_tree_sitter()
    from graphcode.phases.ast_parser import parse_file

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
    _require_tree_sitter()
    from graphcode.phases.ast_parser import parse_file

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


def test_parse_python_docstrings():
    _require_tree_sitter()
    from graphcode.phases.ast_parser import parse_file

    root = _write_tmp({"doc.py": '''
        def documented():
            """This is a docstring."""
            pass

        class MyClass:
            """Class docstring."""
            def method(self):
                """Method docstring."""
                pass
    '''})
    pf = parse_file(str(Path(root) / "doc.py"))
    assert pf is not None
    by_name = {s.name: s for s in pf.symbols}
    assert "This is a docstring." in by_name["documented"].docstring
    assert "Class docstring." in by_name["MyClass"].docstring
    assert "Method docstring." in by_name["MyClass.method"].docstring


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def test_graph_build_nodes_and_edges():
    _require_tree_sitter()
    from graphcode.phases.ast_parser import parse_file
    from graphcode.phases.graph_builder import build_graph
    from graphcode.models import NodeType, EdgeType

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


def test_graph_no_duplicate_edges():
    from graphcode.graph.code_graph import CodeGraph
    from graphcode.models import NodeType, EdgeType, SymbolNode, SymbolEdge

    g = CodeGraph()
    g.add_node(SymbolNode(id="a", name="a", node_type=NodeType.FUNCTION, file_path="f.py"))
    g.add_node(SymbolNode(id="b", name="b", node_type=NodeType.FUNCTION, file_path="f.py"))
    g.add_edge(SymbolEdge(source_id="a", target_id="b", edge_type=EdgeType.CALLS))
    g.add_edge(SymbolEdge(source_id="a", target_id="b", edge_type=EdgeType.CALLS))
    assert g.edge_count() == 1


def test_graph_call_resolution_prefers_same_file():
    _require_tree_sitter()
    from graphcode.phases.ast_parser import parse_file
    from graphcode.phases.graph_builder import build_graph
    from graphcode.models import EdgeType

    root = _write_tmp({
        "a.py": """
            def helper():
                pass

            def main():
                helper()
        """,
        "b.py": """
            def helper():
                pass
        """,
    })
    pf_a = parse_file(str(Path(root) / "a.py"))
    pf_b = parse_file(str(Path(root) / "b.py"))
    assert pf_a and pf_b
    graph = build_graph([pf_a, pf_b])

    main_id = f"{str(Path(root) / 'a.py')}::main"
    helper_a_id = f"{str(Path(root) / 'a.py')}::helper"
    helper_b_id = f"{str(Path(root) / 'b.py')}::helper"

    assert graph.has_edge(main_id, helper_a_id, EdgeType.CALLS)
    assert not graph.has_edge(main_id, helper_b_id, EdgeType.CALLS)


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
# Resolver
# ---------------------------------------------------------------------------

def test_resolver_upgrades_and_cleans_placeholders():
    _require_tree_sitter()
    from graphcode.phases.ast_parser import parse_file
    from graphcode.phases.graph_builder import build_graph
    from graphcode.phases.resolver import resolve_imports
    from graphcode.models import NodeType, EdgeType

    root = _write_tmp({
        "main.py": """
            from utils import helper
        """,
        "utils.py": """
            def helper():
                pass
        """,
    })
    pf_main = parse_file(str(Path(root) / "main.py"))
    pf_utils = parse_file(str(Path(root) / "utils.py"))
    assert pf_main and pf_utils
    graph = build_graph([pf_main, pf_utils])
    graph = resolve_imports(graph, root)

    main_path = str(Path(root) / "main.py")
    utils_path = str(Path(root) / "utils.py")
    assert graph.has_edge(main_path, utils_path, EdgeType.IMPORTS)
    module_nodes = graph.nodes(NodeType.MODULE)
    for mid in module_nodes:
        assert graph.in_degree(mid) == 0 or (graph.get_node_data(mid) or {}).get("name") != "utils"


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------

def test_hybrid_index_bm25_search():
    from graphcode.graph.code_graph import CodeGraph
    from graphcode.phases.indexer import HybridIndex
    from graphcode.models import NodeType, SymbolNode

    g = CodeGraph()
    g.add_node(SymbolNode(id="auth", name="authenticate_user", node_type=NodeType.FUNCTION, file_path="auth.py", signature="def authenticate_user(username, password)"))
    g.add_node(SymbolNode(id="db", name="connect_database", node_type=NodeType.FUNCTION, file_path="db.py", signature="def connect_database(url)"))

    idx = HybridIndex(g)
    idx.build(use_semantic=False)
    results = idx.search("authenticate", top_k=5)
    assert len(results) >= 1
    assert results[0].name == "authenticate_user"


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def test_graph_save_load(tmp_path):
    from graphcode.graph.code_graph import CodeGraph
    from graphcode.models import NodeType, EdgeType, SymbolNode, SymbolEdge

    g = CodeGraph()
    g.add_node(SymbolNode(id="a", name="a", node_type=NodeType.FUNCTION, file_path="f.py"))
    g.add_node(SymbolNode(id="b", name="b", node_type=NodeType.FUNCTION, file_path="f.py"))
    g.add_edge(SymbolEdge(source_id="a", target_id="b", edge_type=EdgeType.CALLS))

    path = str(tmp_path / "graph.json")
    g.save(path)

    loaded = CodeGraph.load(path)
    assert loaded.node_count() == 2
    assert loaded.edge_count() == 1
    assert loaded.has_edge("a", "b")


# ---------------------------------------------------------------------------
# Clustering fallback
# ---------------------------------------------------------------------------

def test_cluster_fallback():
    from graphcode.phases.clustering import cluster_graph

    from graphcode.graph.code_graph import CodeGraph
    from graphcode.models import NodeType, EdgeType, SymbolNode, SymbolEdge

    g = CodeGraph()
    for i in range(5):
        g.add_node(SymbolNode(id=f"sym{i}", name=f"sym{i}", node_type=NodeType.FUNCTION, file_path=f"f{i%2}.py"))
    g.add_edge(SymbolEdge(source_id="sym0", target_id="sym1", edge_type=EdgeType.CALLS))
    g.add_edge(SymbolEdge(source_id="sym2", target_id="sym3", edge_type=EdgeType.CALLS))

    clusters = cluster_graph(g)
    assert len(clusters) >= 1
    total_nodes = sum(len(c.node_ids) for c in clusters)
    assert total_nodes == 5
