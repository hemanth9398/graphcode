"""
Phase 6: Hybrid search index — BM25 + semantic embeddings + RRF fusion.

BM25 catches exact keyword matches; sentence-transformers catch synonyms and
intent. Reciprocal Rank Fusion (RRF) merges both ranked lists without needing
calibrated scores.

sentence-transformers is optional: if unavailable the index falls back to
BM25-only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from graphcode.graph.code_graph import CodeGraph


@dataclass
class SearchResult:
    node_id: str
    name: str
    file_path: str
    node_type: str
    score: float          # RRF combined score
    bm25_rank: int = -1
    semantic_rank: int = -1
    line_start: int = 0


class HybridIndex:
    def __init__(self, graph: CodeGraph) -> None:
        self._graph = graph
        self._node_ids: list[str] = []
        self._docs: list[str] = []
        self._bm25: object | None = None
        self._embedder: object | None = None
        self._embeddings: np.ndarray | None = None
        self._built = False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, use_semantic: bool = True) -> None:
        self._node_ids = list(self._graph.nodes())
        self._docs = []

        for nid in self._node_ids:
            d = self._graph.get_node_data(nid) or {}
            doc = " ".join(filter(None, [
                d.get("name", ""),
                d.get("node_type", ""),
                d.get("signature", ""),
                d.get("docstring", ""),
                (d.get("file_path", "") or "").rsplit("/", 1)[-1],
            ]))
            self._docs.append(doc)

        # BM25
        try:
            from rank_bm25 import BM25Okapi
            tokenised = [doc.lower().split() for doc in self._docs]
            self._bm25 = BM25Okapi(tokenised)
        except ImportError:
            self._bm25 = None

        # Semantic
        if use_semantic:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
                self._embeddings = self._embedder.encode(  # type: ignore[union-attr]
                    self._docs, convert_to_numpy=True, show_progress_bar=False,
                    batch_size=64,
                )
            except (ImportError, Exception):
                self._embedder = None
                self._embeddings = None

        self._built = True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        if not self._built:
            self.build()
        if not self._node_ids:
            return []

        n = len(self._node_ids)
        rrf_scores = [0.0] * n
        ranked_bm25: list[int] = []
        ranked_sem: list[int] = []

        # BM25
        if self._bm25 is not None:
            raw = list(self._bm25.get_scores(query.lower().split()))  # type: ignore[union-attr]
            ranked_bm25 = sorted(range(n), key=lambda i: raw[i], reverse=True)
            for rank, idx in enumerate(ranked_bm25):
                rrf_scores[idx] += 1.0 / (60 + rank + 1)

        # Semantic cosine similarity
        if self._embedder is not None and self._embeddings is not None:
            q_emb = self._embedder.encode([query], convert_to_numpy=True)  # type: ignore[union-attr]
            norms = np.linalg.norm(self._embeddings, axis=1)
            q_norm = float(np.linalg.norm(q_emb))
            sims = (self._embeddings @ q_emb.T).flatten() / (norms * q_norm + 1e-9)
            ranked_sem = sorted(range(n), key=lambda i: float(sims[i]), reverse=True)
            for rank, idx in enumerate(ranked_sem):
                rrf_scores[idx] += 1.0 / (60 + rank + 1)

        top_indices = sorted(range(n), key=lambda i: rrf_scores[i], reverse=True)[:top_k]

        results: list[SearchResult] = []
        for idx in top_indices:
            if rrf_scores[idx] == 0.0:
                break
            nid = self._node_ids[idx]
            d = self._graph.get_node_data(nid) or {}
            results.append(SearchResult(
                node_id=nid,
                name=d.get("name", nid),
                file_path=d.get("file_path", ""),
                node_type=d.get("node_type", ""),
                score=rrf_scores[idx],
                bm25_rank=ranked_bm25.index(idx) if ranked_bm25 else -1,
                semantic_rank=ranked_sem.index(idx) if ranked_sem else -1,
                line_start=d.get("line_start", 0),
            ))

        return results
