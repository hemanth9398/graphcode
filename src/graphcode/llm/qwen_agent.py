"""
QwenAgent: graph-aware LLM wrapper for Qwen2.5-Coder.

Communicates with a locally-running Ollama instance by default.
Falls back to HuggingFace transformers for offline / embedded use.

The agent understands the CodeGraph's structure and can:
  - Analyse execution routes
  - Describe code clusters
  - Emulate symbolic execution
  - Answer freeform questions grounded in the graph
"""
from __future__ import annotations

import json
import urllib.request
from typing import Any

from graphcode.graph.code_graph import CodeGraph
from graphcode.models import Cluster, Route


class QwenAgent:
    def __init__(
        self,
        model: str = "qwen2.5-coder:1.5b",
        use_ollama: bool = True,
        ollama_url: str = "http://localhost:11434",
    ) -> None:
        self._model = model
        self._use_ollama = use_ollama
        self._ollama_url = ollama_url.rstrip("/")
        self._pipeline: Any = None
        if not use_ollama:
            self._load_local()

    # ------------------------------------------------------------------
    # Backend communication
    # ------------------------------------------------------------------

    def _load_local(self) -> None:
        try:
            from transformers import pipeline
            import torch
            self._pipeline = pipeline(
                "text-generation",
                model=self._model,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        except (ImportError, Exception) as exc:
            self._pipeline = None
            print(f"[QwenAgent] Local model unavailable: {exc}")

    def _call_ollama(self, prompt: str, max_tokens: int = 512) -> str:
        payload = json.dumps({
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.2},
        }).encode()
        try:
            req = urllib.request.Request(
                f"{self._ollama_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode())["response"]
        except Exception as exc:
            return f"[Ollama error: {exc}]"

    def _call_local(self, prompt: str, max_tokens: int = 512) -> str:
        if not self._pipeline:
            return "[Model not loaded — run with Ollama or install transformers+torch]"
        out = self._pipeline(prompt, max_new_tokens=max_tokens, do_sample=False)
        return out[0]["generated_text"][len(prompt):]

    def call(self, prompt: str, max_tokens: int = 512) -> str:
        """Send a raw prompt and return the model's response."""
        if self._use_ollama:
            return self._call_ollama(prompt, max_tokens)
        return self._call_local(prompt, max_tokens)

    # ------------------------------------------------------------------
    # Graph serialisation helpers
    # ------------------------------------------------------------------

    def _serialise_subgraph(
        self, graph: CodeGraph, node_ids: list[str], max_nodes: int = 40
    ) -> str:
        lines: list[str] = []
        for nid in node_ids[:max_nodes]:
            d = graph.get_node_data(nid) or {}
            name = d.get("name", nid)
            ntype = d.get("node_type", "?")
            fp = (d.get("file_path", "") or "").rsplit("/", 1)[-1]
            sig = d.get("signature", "")
            line_info = f"  [{ntype}] {name}"
            if fp:
                line_info += f"  ({fp})"
            if sig:
                line_info += f"\n    sig: {sig}"
            lines.append(line_info)
            # Show outgoing edges
            for _, dst, et in graph.edges(source=nid)[:5]:
                dst_d = graph.get_node_data(dst) or {}
                lines.append(f"    --{et}--> {dst_d.get('name', dst)}")
        if len(node_ids) > max_nodes:
            lines.append(f"  ... ({len(node_ids) - max_nodes} more nodes)")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # High-level analysis APIs
    # ------------------------------------------------------------------

    def analyse_route(self, graph: CodeGraph, route: Route) -> str:
        ctx = self._serialise_subgraph(graph, route.nodes)
        prompt = f"""You are a code intelligence assistant analysing a code graph.

ENTRY POINT: {route.entry_name}
TERRITORY SIZE: {len(route.nodes)} nodes, max depth {route.depth}

GRAPH STRUCTURE:
{ctx}

Provide a concise analysis covering:
1. Purpose — what does this entry point do?
2. Key call chain — the 3-5 most important steps in order
3. External dependencies — what modules/files does it rely on?
4. Risks — circular deps, overly deep chains, missing deps

ANALYSIS:"""
        return self.call(prompt, max_tokens=600)

    def analyse_cluster(self, graph: CodeGraph, cluster: Cluster) -> str:
        ctx = self._serialise_subgraph(graph, cluster.node_ids)
        prompt = f"""You are a code intelligence assistant analysing a code cluster.

CLUSTER: {cluster.label}  (cohesion {cluster.cohesion_score:.2f}, {len(cluster.node_ids)} nodes)

GRAPH STRUCTURE:
{ctx}

Describe this cluster:
1. Primary responsibility — one sentence
2. Cohesion assessment — is this cluster well-focused or a grab-bag?
3. Suggested module/package name for this cluster
4. Any symbols that don't belong here

CLUSTER ANALYSIS:"""
        return self.call(prompt, max_tokens=400)

    def emulate_execution(self, graph: CodeGraph, start_node_id: str) -> str:
        """Symbolically trace execution from *start_node_id* through the graph."""
        from graphcode.graph.traversal import dfs
        path = dfs(graph, start_node_id, max_depth=10)
        ctx = self._serialise_subgraph(graph, [n for n, _ in path])
        start_d = graph.get_node_data(start_node_id) or {}

        prompt = f"""You are symbolically emulating a code execution path through a graph.

START: {start_d.get('name', start_node_id)}  ({start_d.get('node_type', '?')})
FILE:  {start_d.get('file_path', '')}
SIGNATURE: {start_d.get('signature', '')}

DFS EXECUTION TRACE:
{ctx}

Trace the symbolic execution:
1. What does calling this function/method do?
2. What data does it read, transform, or write?
3. What side effects propagate through the call chain?
4. Under what conditions could this fail?

EMULATION:"""
        return self.call(prompt, max_tokens=600)

    def answer_query(
        self,
        graph: CodeGraph,
        query: str,
        context_nodes: list[str] | None = None,
    ) -> str:
        """Answer a freeform user question, grounding the response in graph context."""
        stats = graph.stats()
        ctx = self._serialise_subgraph(graph, context_nodes or []) if context_nodes else ""
        prompt = f"""You are a code intelligence assistant with access to a code graph.

GRAPH STATS: {stats}

RELEVANT CONTEXT:
{ctx if ctx else "(no specific nodes provided — use general graph knowledge)"}

USER QUESTION: {query}

ANSWER:"""
        return self.call(prompt, max_tokens=512)

    def explain_path(
        self, graph: CodeGraph, path: list[str]
    ) -> str:
        """Explain what a specific graph path (sequence of node IDs) represents."""
        steps: list[str] = []
        for i, nid in enumerate(path):
            d = graph.get_node_data(nid) or {}
            steps.append(f"  {i+1}. [{d.get('node_type','?')}] {d.get('name', nid)}")
        prompt = f"""You are a code intelligence assistant.

The following path was found in a code graph:
{chr(10).join(steps)}

Explain:
1. What this execution/dependency path represents
2. Why these symbols are connected
3. Whether this path indicates a design issue

PATH EXPLANATION:"""
        return self.call(prompt, max_tokens=400)
