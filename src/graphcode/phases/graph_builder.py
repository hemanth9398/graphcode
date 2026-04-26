"""
Phase 3: Build the CodeGraph from a collection of ParsedFiles.

Node creation order:
  1. FILE node for every parsed file
  2. SYMBOL nodes for every extracted symbol
  3. FILE --defines--> SYMBOL  edges
  4. CLASS --contains--> METHOD  edges (intra-file)
  5. FILE --imports--> MODULE   edges (unresolved; Phase 4 resolves them)
  6. CALLER --calls--> CALLEE   edges (name-matched across all files)
"""
from __future__ import annotations

from graphcode.graph.code_graph import CodeGraph
from graphcode.models import EdgeType, NodeType, SymbolEdge, SymbolNode
from graphcode.phases.ast_parser import ParsedFile


def build_graph(parsed_files: list[ParsedFile]) -> CodeGraph:
    graph = CodeGraph()

    # ── Pass 1: add all file + symbol nodes ─────────────────────────────
    # Build a name → [node_id] index for call resolution in pass 2.
    name_index: dict[str, list[str]] = {}

    for pf in parsed_files:
        # File node
        graph.add_node(SymbolNode(
            id=pf.path,
            name=pf.path.rsplit("/", 1)[-1],
            node_type=NodeType.FILE,
            file_path=pf.path,
            language=pf.language,
        ))

        for sym in pf.symbols:
            graph.add_node(sym)
            # Index by bare name (last segment after dot) for fuzzy call matching
            bare = sym.name.split(".")[-1]
            name_index.setdefault(bare, []).append(sym.id)
            name_index.setdefault(sym.name, []).append(sym.id)

        # Unresolved module placeholder nodes
        for imp in pf.imports:
            module = imp.get("module", "")
            if not module:
                continue
            mod_id = f"module::{module}"
            if not graph.has_node(mod_id):
                graph.add_node(SymbolNode(
                    id=mod_id,
                    name=module,
                    node_type=NodeType.MODULE,
                    file_path="",
                    language=pf.language,
                ))

    # ── Pass 2: add structural edges ────────────────────────────────────
    for pf in parsed_files:
        # FILE --defines--> SYMBOL
        for sym in pf.symbols:
            graph.add_edge(SymbolEdge(
                source_id=pf.path,
                target_id=sym.id,
                edge_type=EdgeType.DEFINES,
            ))

            # CLASS --contains--> METHOD
            if sym.node_type == NodeType.METHOD and "." in sym.name:
                class_name = sym.name.rsplit(".", 1)[0]
                class_id = f"{pf.path}::{class_name}"
                if graph.has_node(class_id):
                    graph.add_edge(SymbolEdge(
                        source_id=class_id,
                        target_id=sym.id,
                        edge_type=EdgeType.CONTAINS,
                    ))

        # FILE --imports--> MODULE (unresolved placeholder)
        for imp in pf.imports:
            module = imp.get("module", "")
            if not module:
                continue
            mod_id = f"module::{module}"
            if graph.has_node(mod_id) and not graph.has_edge(pf.path, mod_id, EdgeType.IMPORTS):
                graph.add_edge(SymbolEdge(
                    source_id=pf.path,
                    target_id=mod_id,
                    edge_type=EdgeType.IMPORTS,
                ))

        # CALLER --calls--> CALLEE (name-matched)
        for ref in pf.call_refs:
            if not graph.has_node(ref.caller_id):
                continue
            callee_name = ref.callee_name
            candidates = name_index.get(callee_name) or name_index.get(
                callee_name.split(".")[-1], []
            )
            for callee_id in candidates:
                if callee_id == ref.caller_id:
                    continue  # skip self-calls
                if not graph.has_edge(ref.caller_id, callee_id, EdgeType.CALLS):
                    graph.add_edge(SymbolEdge(
                        source_id=ref.caller_id,
                        target_id=callee_id,
                        edge_type=EdgeType.CALLS,
                        line=ref.line,
                    ))

    return graph
