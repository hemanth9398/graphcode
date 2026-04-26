"""
LadybugDB — thin wrapper around KuzuDB (formerly KuzuDB, now the embedded
graph database with native vector support that GraphCode calls "LadybugDB").

Persists the CodeGraph as a property graph: Symbol nodes + Relationship edges.
Falls back silently when kuzu is not installed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from graphcode.graph.code_graph import CodeGraph


class LadybugDB:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._db: Any = None
        self._conn: Any = None
        self._available = False
        self._init()

    def _init(self) -> None:
        try:
            import kuzu
            Path(self._db_path).mkdir(parents=True, exist_ok=True)
            self._db = kuzu.Database(self._db_path)
            self._conn = kuzu.Connection(self._db)
            self._create_schema()
            self._available = True
        except (ImportError, Exception) as exc:
            print(f"[LadybugDB] kuzu unavailable ({exc}); graph will not be persisted.")

    def _create_schema(self) -> None:
        if not self._conn:
            return
        self._exec("""
            CREATE NODE TABLE IF NOT EXISTS Symbol (
                id      STRING PRIMARY KEY,
                name    STRING,
                node_type STRING,
                file_path STRING,
                line_start INT64,
                line_end  INT64,
                signature STRING,
                language  STRING
            )
        """)
        self._exec("""
            CREATE REL TABLE IF NOT EXISTS Rel (
                FROM Symbol TO Symbol,
                edge_type STRING,
                line      INT64
            )
        """)

    def _exec(self, cypher: str, params: dict | None = None) -> Any:
        if not self._conn:
            return None
        try:
            return self._conn.execute(cypher, params or {})
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def persist_graph(self, graph: CodeGraph) -> None:
        if not self._available:
            return

        for nid in graph.nodes():
            d = graph.get_node_data(nid) or {}
            self._exec(
                "MERGE (s:Symbol {id: $id}) "
                "SET s.name=$name, s.node_type=$nt, s.file_path=$fp, "
                "    s.line_start=$ls, s.line_end=$le, s.signature=$sig, s.language=$lang",
                {
                    "id": nid,
                    "name": d.get("name", ""),
                    "nt": d.get("node_type", ""),
                    "fp": d.get("file_path", ""),
                    "ls": d.get("line_start", 0),
                    "le": d.get("line_end", 0),
                    "sig": d.get("signature", ""),
                    "lang": d.get("language", ""),
                },
            )

        for src, dst, et in graph.edges():
            self._exec(
                "MATCH (a:Symbol {id: $src}), (b:Symbol {id: $dst}) "
                "CREATE (a)-[:Rel {edge_type: $et, line: 0}]->(b)",
                {"src": src, "dst": dst, "et": et},
            )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, cypher: str, params: dict | None = None) -> list[Any]:
        if not self._available:
            return []
        try:
            result = self._conn.execute(cypher, params or {})
            rows: list[Any] = []
            while result.has_next():
                rows.append(result.get_next())
            return rows
        except Exception:
            return []

    def find_symbol(self, name: str) -> list[dict]:
        rows = self.query(
            "MATCH (s:Symbol) WHERE s.name CONTAINS $name RETURN s.id, s.name, s.node_type, s.file_path",
            {"name": name},
        )
        return [{"id": r[0], "name": r[1], "node_type": r[2], "file_path": r[3]} for r in rows]

    def neighbours(self, symbol_id: str) -> list[dict]:
        rows = self.query(
            "MATCH (a:Symbol {id: $id})-[r:Rel]->(b:Symbol) "
            "RETURN b.id, b.name, r.edge_type",
            {"id": symbol_id},
        )
        return [{"id": r[0], "name": r[1], "edge_type": r[2]} for r in rows]

    def close(self) -> None:
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
