"""
GraphCode CLI — four commands:

  graphcode analyse  <path>           full pipeline + report
  graphcode search   <path> <query>   hybrid BM25+semantic search
  graphcode route    <path>           show execution routes (+ Qwen analysis)
  graphcode emulate  <path> <symbol>  symbolically emulate a symbol via Qwen
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from graphcode.pipeline import GraphCodeSession, run_pipeline

app = typer.Typer(
    name="graphcode",
    help="AST-driven code graph intelligence with Qwen emulation.",
    add_completion=False,
)
console = Console()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _stats_table(session: GraphCodeSession) -> Table:
    table = Table(title="GraphCode  —  Analysis Results", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan", min_width=22)
    table.add_column("Value", style="green")

    if session.structure:
        table.add_row("Files found", str(session.structure.file_count))
        by_ext = session.structure.files_by_extension()
        for ext, files in sorted(by_ext.items())[:6]:
            table.add_row(f"  {ext}", str(len(files)))

    table.add_row("Files parsed", str(len(session.parsed_files)))
    total_syms = sum(len(p.symbols) for p in session.parsed_files)
    table.add_row("Symbols extracted", str(total_syms))

    if session.graph:
        stats = session.graph.stats()
        table.add_row("Graph nodes", str(stats["nodes"]))
        table.add_row("Graph edges", str(stats["edges"]))
        for nt, cnt in sorted(stats.get("node_types", {}).items()):
            table.add_row(f"  {nt}", str(cnt))

    table.add_row("Routes", str(len(session.routes)))
    table.add_row("Clusters", str(len(session.clusters)))
    return table


# ---------------------------------------------------------------------------
# analyse
# ---------------------------------------------------------------------------

@app.command("analyse")
def cmd_analyse(
    path: str = typer.Argument(".", help="Codebase root"),
    db: Optional[str] = typer.Option(None, "--db", help="LadybugDB output path"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Skip Qwen analysis"),
    no_semantic: bool = typer.Option(False, "--no-semantic", help="Skip semantic embeddings"),
    model: str = typer.Option("qwen2.5-coder:1.5b", "--model", "-m", help="Qwen Ollama model"),
    persist: bool = typer.Option(False, "--persist", help="Persist graph to LadybugDB"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Write JSON results"),
) -> None:
    """Run the full GraphCode pipeline and print a summary report."""
    root = str(Path(path).resolve())
    console.print(Panel(f"[bold cyan]GraphCode[/]  analysing  [yellow]{root}[/]", expand=False))

    session = run_pipeline(
        root,
        db_path=db,
        use_llm=not no_llm,
        llm_model=model,
        use_semantic=not no_semantic,
        persist=persist,
    )

    console.print()
    console.print(_stats_table(session))

    # Routes
    if session.routes:
        console.print("\n[bold]Top Execution Routes[/]")
        for r in session.routes[:6]:
            console.print(
                f"  [cyan]{r.entry_name}[/]"
                f"  →  {len(r.nodes)} nodes  (depth {r.depth})"
            )

    # Clusters
    if session.clusters:
        console.print("\n[bold]Top Clusters[/]")
        for c in session.clusters[:6]:
            console.print(
                f"  [magenta]#{c.id}[/] [{c.label}]"
                f"  {len(c.node_ids)} nodes  cohesion {c.cohesion_score:.2f}"
            )

    # Emulation report
    if session.emulation_report:
        rpt = session.emulation_report
        console.print(Panel(rpt.summary, title="[bold]Emulation Report[/]", expand=False))
        for issue in rpt.issues[:10]:
            icon = {"error": "✗", "warning": "⚠", "info": "ℹ"}.get(issue.severity, "·")
            color = {"error": "red", "warning": "yellow", "info": "blue"}.get(issue.severity, "white")
            console.print(f"  [{color}]{icon}[/] {issue.description}")
            if issue.suggestion:
                console.print(f"      [dim]{issue.suggestion}[/]")

        if rpt.llm_analysis:
            console.print(Panel(
                rpt.llm_analysis[:800],
                title="[bold green]Qwen  Graph Assessment[/]",
                expand=False,
            ))

    # JSON output
    if output:
        result = {
            "root": root,
            "file_count": session.structure.file_count if session.structure else 0,
            "symbols": sum(len(p.symbols) for p in session.parsed_files),
            "graph": session.graph.stats() if session.graph else {},
            "routes": [
                {"entry": r.entry_name, "nodes": len(r.nodes), "depth": r.depth}
                for r in session.routes
            ],
            "clusters": [
                {"id": c.id, "label": c.label, "size": len(c.node_ids), "cohesion": c.cohesion_score}
                for c in session.clusters
            ],
            "issues": [
                {"kind": i.kind, "severity": i.severity, "description": i.description,
                 "suggestion": i.suggestion}
                for i in (session.emulation_report.issues if session.emulation_report else [])
            ],
        }
        Path(output).write_text(json.dumps(result, indent=2))
        console.print(f"\n[green]Results written to {output}[/]")


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

@app.command("search")
def cmd_search(
    path: str = typer.Argument(".", help="Codebase root"),
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Results to return"),
    no_semantic: bool = typer.Option(False, "--no-semantic"),
) -> None:
    """Hybrid BM25 + semantic search across all code symbols."""
    root = str(Path(path).resolve())
    console.print(f"Indexing [yellow]{root}[/] …")

    session = run_pipeline(root, use_llm=False, use_semantic=not no_semantic, quiet=True)
    if not session.index:
        console.print("[red]Index not built[/]")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]Search:[/]  {query}\n")
    results = session.index.search(query, top_k=top_k)

    table = Table(show_header=True, header_style="bold")
    table.add_column("Score", style="yellow", justify="right")
    table.add_column("BM25#", justify="right")
    table.add_column("Sem#", justify="right")
    table.add_column("Type", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("File")
    table.add_column("Line", justify="right")

    for r in results:
        table.add_row(
            f"{r.score:.4f}",
            str(r.bm25_rank) if r.bm25_rank >= 0 else "-",
            str(r.semantic_rank) if r.semantic_rank >= 0 else "-",
            r.node_type,
            r.name,
            r.file_path.rsplit("/", 1)[-1] if r.file_path else "",
            str(r.line_start) if r.line_start else "-",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# route
# ---------------------------------------------------------------------------

@app.command("route")
def cmd_route(
    path: str = typer.Argument(".", help="Codebase root"),
    entry: Optional[str] = typer.Option(None, "--entry", "-e", help="Filter by entry name"),
    ask: Optional[str] = typer.Option(None, "--ask", help="Ask Qwen about this route"),
    model: str = typer.Option("qwen2.5-coder:1.5b", "--model", "-m"),
    max_show: int = typer.Option(20, "--max-show", help="Max nodes to display per route"),
) -> None:
    """Show execution routes traced from entry points via DFS."""
    root = str(Path(path).resolve())
    session = run_pipeline(root, use_llm=False, quiet=True)

    if not session.routes:
        console.print("[yellow]No routes found.[/]")
        raise typer.Exit(0)

    routes_to_show = (
        [r for r in session.routes if entry and (entry in r.entry_name or entry in r.entry_id)]
        or session.routes
    )[:5]

    for route in routes_to_show:
        tree = Tree(
            f"[bold cyan]{route.entry_name}[/]"
            f"  depth={route.depth}  nodes={len(route.nodes)}"
        )
        for nid in route.nodes[:max_show]:
            d = (session.graph.get_node_data(nid) if session.graph else None) or {}
            tree.add(f"[{d.get('node_type','?')}] {d.get('name', nid)}")
        if len(route.nodes) > max_show:
            tree.add(f"[dim]… {len(route.nodes) - max_show} more[/]")
        console.print(tree)

        if ask and session.graph:
            from graphcode.llm.qwen_agent import QwenAgent
            agent = QwenAgent(model=model)
            analysis = agent.analyse_route(session.graph, route)
            console.print(Panel(analysis, title="[bold green]Qwen Route Analysis[/]", expand=False))


# ---------------------------------------------------------------------------
# emulate
# ---------------------------------------------------------------------------

@app.command("emulate")
def cmd_emulate(
    path: str = typer.Argument(".", help="Codebase root"),
    symbol: str = typer.Argument(..., help="Symbol name substring to emulate"),
    model: str = typer.Option("qwen2.5-coder:1.5b", "--model", "-m"),
) -> None:
    """Symbolically emulate a symbol's execution through the graph using Qwen."""
    root = str(Path(path).resolve())
    session = run_pipeline(root, use_llm=False, quiet=True)

    if not session.graph:
        console.print("[red]Graph not built[/]")
        raise typer.Exit(1)

    matching = [n for n in session.graph.nodes() if symbol in n]
    if not matching:
        console.print(f"[red]No symbol matching '{symbol}'[/]")
        raise typer.Exit(1)

    # Pick best match (prefer exact name match over file-path substring)
    best = next(
        (n for n in matching
         if (session.graph.get_node_data(n) or {}).get("name", "").endswith(symbol)),
        matching[0],
    )

    from graphcode.llm.qwen_agent import QwenAgent
    agent = QwenAgent(model=model)
    result = agent.emulate_execution(session.graph, best)
    d = session.graph.get_node_data(best) or {}
    console.print(Panel(
        result,
        title=f"[bold green]Emulation:  {d.get('name', best)}[/]",
        expand=False,
    ))


if __name__ == "__main__":
    app()
