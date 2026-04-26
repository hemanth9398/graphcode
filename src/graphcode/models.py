"""Core data models shared across all pipeline phases."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeType(str, Enum):
    FILE = "file"
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    INTERFACE = "interface"
    VARIABLE = "variable"
    MODULE = "module"      # unresolved external module placeholder
    IMPORT = "import"


class EdgeType(str, Enum):
    IMPORTS = "imports"        # file ──imports──> module/file
    CALLS = "calls"            # function ──calls──> function
    EXTENDS = "extends"        # class ──extends──> class
    IMPLEMENTS = "implements"  # class ──implements──> interface
    DEFINES = "defines"        # file/class ──defines──> symbol
    REFERENCES = "references"  # function ──references──> variable/type
    CONTAINS = "contains"      # class ──contains──> method
    EXPORTS = "exports"        # file ──exports──> symbol


@dataclass
class SymbolNode:
    id: str           # globally unique: "abs/file/path.py::QualifiedName"
    name: str         # qualified name ("MyClass.my_method" or "my_func")
    node_type: NodeType
    file_path: str
    line_start: int = 0
    line_end: int = 0
    signature: str = ""
    docstring: str = ""
    language: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SymbolEdge:
    source_id: str
    target_id: str
    edge_type: EdgeType
    line: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CallRef:
    """Unresolved call reference extracted from AST — resolved in graph_builder."""
    caller_id: str    # ID of the calling symbol
    callee_name: str  # raw name as written in source (may include dots)
    line: int = 0


@dataclass
class Route:
    """Execution territory of one entry point traced by DFS."""
    entry_id: str
    entry_name: str
    nodes: list[str]                       # DFS-ordered node IDs
    edges: list[tuple[str, str, str]]      # (src, dst, edge_type) within territory
    depth: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Cluster:
    """A community of related symbols found by Leiden detection."""
    id: int
    node_ids: list[str]
    cohesion_score: float = 0.0
    label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
