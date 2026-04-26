"""
GraphCode pipeline orchestrator — runs all 6 phases in order.

Phase 1  Structure    — walk file tree
Phase 2  AST          — Tree-sitter symbol extraction + call refs
Phase 3  Graph        — build CodeGraph from ASTs; extract routes
Phase 4  Resolve      — cross-file import resolution
Phase 5  Cluster      — Leiden community detection
Phase 6  Index        — hybrid BM25 + semantic search

Optional:  LLM emulation (Qwen), LadybugDB persistence
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from graphcode.graph.code_graph import CodeGraph
from graphcode.graph.routes import Route, extract_routes
from graphcode.llm.emulator import EmulationReport, GraphEmulator
from graphcode.llm.qwen_agent import QwenAgent
from graphcode.models import Cluster
from graphcode.phases.ast_parser import ParsedFile, parse_file
from graphcode.phases.clustering import cluster_graph
from graphcode.phases.graph_builder import build_graph
from graphcode.phases.indexer import HybridIndex
from graphcode.phases.resolver import resolve_imports
from graphcode.phases.structure import StructureMap, walk_tree
from graphcode.storage.ladybug_db import LadybugDB

console = Console()


@dataclass
class GraphCodeSession:
    root: str
    structure: StructureMap | None = None
    parsed_files: list[ParsedFile] = field(default_factory=list)
    graph: CodeGraph | None = None
    routes: list[Route] = field(default_factory=list)
    clusters: list[Cluster] = field(default_factory=list)
    index: HybridIndex | None = None
    db: LadybugDB | None = None
    agent: QwenAgent | None = None
    emulation_report: EmulationReport | None = None


def run_pipeline(
    root: str,
    *,
    db_path: str | None = None,
    use_llm: bool = True,
    llm_model: str = "qwen2.5-coder:1.5b",
    use_ollama: bool = True,
    use_semantic: bool = True,
    persist: bool = False,
    quiet: bool = False,
) -> GraphCodeSession:
    """
    Run the full GraphCode pipeline and return a populated session.

    All options are keyword-only:
      db_path      — LadybugDB directory (required when persist=True)
      use_llm      — run Qwen emulation phase
      llm_model    — Ollama model tag (default qwen2.5-coder:1.5b)
      use_ollama   — use Ollama HTTP API vs. local transformers
      use_semantic — build sentence-transformer embeddings
      persist      — write graph to LadybugDB
      quiet        — suppress progress output
    """
    session = GraphCodeSession(root=root)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        console=console,
        disable=quiet,
    ) as progress:

        # ── Phase 1: Structure ────────────────────────────────────────────
        t = progress.add_task("Phase 1  walking file tree …", total=None)
        session.structure = walk_tree(root)
        progress.update(t, description=f"[green]Phase 1 ✓  {session.structure.file_count} files found")
        progress.stop_task(t)

        # ── Phase 2: AST Parsing ──────────────────────────────────────────
        t = progress.add_task("Phase 2  parsing ASTs (Tree-sitter) …", total=None)
        for f in session.structure.files:
            pf = parse_file(f.path)
            if pf:
                session.parsed_files.append(pf)
        total_syms = sum(len(p.symbols) for p in session.parsed_files)
        progress.update(t, description=f"[green]Phase 2 ✓  {len(session.parsed_files)} files · {total_syms} symbols")
        progress.stop_task(t)

        # ── Phase 3a: Graph construction ──────────────────────────────────
        t = progress.add_task("Phase 3  building code graph …", total=None)
        session.graph = build_graph(session.parsed_files)
        s = session.graph.stats()
        progress.update(t, description=f"[green]Phase 3 ✓  {s['nodes']} nodes · {s['edges']} edges")
        progress.stop_task(t)

        # ── Phase 4: Cross-file resolution ────────────────────────────────
        t = progress.add_task("Phase 4  resolving cross-file imports …", total=None)
        session.graph = resolve_imports(session.graph, root)
        progress.update(t, description="[green]Phase 4 ✓  imports resolved")
        progress.stop_task(t)

        # ── Phase 3b: Route extraction (after resolution) ─────────────────
        t = progress.add_task("Phase 3b extracting execution routes …", total=None)
        session.routes = extract_routes(session.graph)
        progress.update(t, description=f"[green]Phase 3b ✓  {len(session.routes)} routes mapped")
        progress.stop_task(t)

        # ── Phase 5: Clustering ───────────────────────────────────────────
        t = progress.add_task("Phase 5  Leiden community clustering …", total=None)
        session.clusters = cluster_graph(session.graph)
        progress.update(t, description=f"[green]Phase 5 ✓  {len(session.clusters)} clusters")
        progress.stop_task(t)

        # ── Phase 6: Hybrid index ─────────────────────────────────────────
        t = progress.add_task("Phase 6  building hybrid BM25+semantic index …", total=None)
        session.index = HybridIndex(session.graph)
        session.index.build(use_semantic=use_semantic)
        mode = "BM25+semantic" if use_semantic and session.index._embedder else "BM25"
        progress.update(t, description=f"[green]Phase 6 ✓  {mode} index ready")
        progress.stop_task(t)

        # ── LLM: Graph emulation ──────────────────────────────────────────
        if use_llm:
            t = progress.add_task("LLM  Qwen graph emulation …", total=None)
            session.agent = QwenAgent(model=llm_model, use_ollama=use_ollama)
            emulator = GraphEmulator(session.graph, session.agent)
            session.emulation_report = emulator.run(use_llm=True)
            n_issues = len(session.emulation_report.issues)
            progress.update(t, description=f"[green]LLM ✓  {n_issues} issues detected")
            progress.stop_task(t)

        # ── Persist to LadybugDB ──────────────────────────────────────────
        if persist:
            _db_path = db_path or f"{root}/.graphcode_db"
            t = progress.add_task("DB  persisting to LadybugDB …", total=None)
            session.db = LadybugDB(_db_path)
            session.db.persist_graph(session.graph)
            progress.update(t, description=f"[green]DB ✓  persisted → {_db_path}")
            progress.stop_task(t)

    return session
