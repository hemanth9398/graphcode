"""
GraphEmulator: self-aware structural analysis + Qwen-driven correction.

The emulator makes the graph introspect itself:
  - Detects structural issues (cycles, orphans, deep chains, missing deps)
  - Generates improvement suggestions
  - Uses QwenAgent to produce a natural-language health report

"Self-awareness" means the system can see its own topology, find gaps,
and recommend corrections — not just passively index.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from graphcode.graph.code_graph import CodeGraph
from graphcode.graph.traversal import dfs, find_cycles, find_entry_points
from graphcode.models import NodeType


# ---------------------------------------------------------------------------
# Issue types
# ---------------------------------------------------------------------------

@dataclass
class GraphIssue:
    kind: str        # cycle | orphan | deep_chain | missing_dep | god_node | dead_file
    severity: str    # error | warning | info
    node_ids: list[str]
    description: str
    suggestion: str = ""


@dataclass
class EmulationReport:
    issues: list[GraphIssue] = field(default_factory=list)
    summary: str = ""
    llm_analysis: str = ""

    @property
    def errors(self) -> list[GraphIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[GraphIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def infos(self) -> list[GraphIssue]:
        return [i for i in self.issues if i.severity == "info"]


# ---------------------------------------------------------------------------
# Emulator
# ---------------------------------------------------------------------------

class GraphEmulator:
    def __init__(self, graph: CodeGraph, agent: object | None = None) -> None:
        self._graph = graph
        self._agent = agent   # QwenAgent | None

    # ── Individual detectors ────────────────────────────────────────────

    def _detect_cycles(self) -> list[GraphIssue]:
        cycles = find_cycles(self._graph)
        issues: list[GraphIssue] = []
        for cycle in cycles[:10]:
            labels = [
                (self._graph.get_node_data(n) or {}).get("name", n)
                for n in cycle[:5]
            ]
            trail = " → ".join(labels) + ("…" if len(cycle) > 5 else "")
            issues.append(GraphIssue(
                kind="cycle",
                severity="warning",
                node_ids=cycle,
                description=f"Circular dependency: {trail}",
                suggestion="Extract shared logic into a neutral module to break the cycle.",
            ))
        return issues

    def _detect_orphans(self) -> list[GraphIssue]:
        issues: list[GraphIssue] = []
        for nid in self._graph.nodes():
            d = self._graph.get_node_data(nid) or {}
            if d.get("node_type") in (NodeType.FILE.value, NodeType.MODULE.value):
                continue  # files/modules can be legitimately unreferenced
            if self._graph.in_degree(nid) == 0 and self._graph.out_degree(nid) == 0:
                issues.append(GraphIssue(
                    kind="orphan",
                    severity="info",
                    node_ids=[nid],
                    description=f"Disconnected symbol: {d.get('name', nid)} in {d.get('file_path','').rsplit('/',1)[-1]}",
                    suggestion="This symbol may be dead code or missing import connections.",
                ))
        return issues[:25]

    def _detect_deep_chains(self, threshold: int = 12) -> list[GraphIssue]:
        issues: list[GraphIssue] = []
        for entry in find_entry_points(self._graph)[:15]:
            path = dfs(self._graph, entry, max_depth=50)
            if not path:
                continue
            max_depth = max(d for _, d in path)
            if max_depth >= threshold:
                name = (self._graph.get_node_data(entry) or {}).get("name", entry)
                issues.append(GraphIssue(
                    kind="deep_chain",
                    severity="warning",
                    node_ids=[entry],
                    description=f"Deep call chain from '{name}': depth {max_depth}",
                    suggestion="Break this into smaller, more focused functions or use an intermediary layer.",
                ))
        return issues

    def _detect_missing_deps(self) -> list[GraphIssue]:
        """Unresolved MODULE placeholder nodes that are actually imported."""
        issues: list[GraphIssue] = []
        for nid in self._graph.nodes():
            d = self._graph.get_node_data(nid) or {}
            if d.get("node_type") != NodeType.MODULE.value:
                continue
            importers = self._graph.in_degree(nid)
            if importers > 0:
                issues.append(GraphIssue(
                    kind="missing_dep",
                    severity="info",
                    node_ids=[nid],
                    description=f"Unresolved module '{d.get('name', nid)}' imported by {importers} file(s)",
                    suggestion="Install the package or check that the import path is correct.",
                ))
        return issues[:20]

    def _detect_god_nodes(self, fan_out_threshold: int = 20) -> list[GraphIssue]:
        """Symbols that call or depend on an unusually large number of others."""
        issues: list[GraphIssue] = []
        for nid in self._graph.nodes():
            out = self._graph.out_degree(nid)
            if out >= fan_out_threshold:
                d = self._graph.get_node_data(nid) or {}
                issues.append(GraphIssue(
                    kind="god_node",
                    severity="warning",
                    node_ids=[nid],
                    description=f"God node '{d.get('name', nid)}' has {out} outgoing edges",
                    suggestion="Consider splitting this into multiple smaller symbols (SRP).",
                ))
        return issues[:10]

    # ── Main entry ────────────────────────────────────────────────────────

    def run(self, use_llm: bool = True) -> EmulationReport:
        report = EmulationReport()

        report.issues += self._detect_cycles()
        report.issues += self._detect_orphans()
        report.issues += self._detect_deep_chains()
        report.issues += self._detect_missing_deps()
        report.issues += self._detect_god_nodes()

        stats = self._graph.stats()
        ne, nw, ni = len(report.errors), len(report.warnings), len(report.infos)
        report.summary = (
            f"Graph: {stats['nodes']} nodes, {stats['edges']} edges — "
            f"{ne} errors · {nw} warnings · {ni} info"
        )

        if use_llm and self._agent is not None:
            issue_lines = "\n".join(
                f"  [{i.severity.upper()}] {i.kind}: {i.description}"
                for i in report.issues[:15]
            )
            prompt = f"""You are a senior software architect reviewing a code graph.

GRAPH STATS:
  Nodes: {stats['nodes']}  Edges: {stats['edges']}
  Node types: {stats.get('node_types', {})}
  Edge types: {stats.get('edge_types', {})}

DETECTED ISSUES:
{issue_lines if issue_lines else "  (none detected)"}

Provide:
1. Overall codebase health (1-5 stars, with justification)
2. Top 3 most impactful improvements
3. One architectural pattern or refactoring that would address multiple issues

ASSESSMENT:"""
            report.llm_analysis = self._agent.call(prompt, max_tokens=700)  # type: ignore[union-attr]

        return report
