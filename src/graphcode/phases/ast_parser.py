"""
Phase 2: Parse source files with Tree-sitter to extract symbols and call references.

Supported languages: Python, JavaScript, TypeScript/TSX.
Other languages fall back gracefully (file node only, no symbols).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from graphcode.models import CallRef, NodeType, SymbolNode

# ---------------------------------------------------------------------------
# Language registry — lazy-load grammars so missing packages don't crash boot
# ---------------------------------------------------------------------------

_LANG_CACHE: dict[str, Any] = {}   # name -> tree_sitter.Language


def _get_language(name: str) -> Any | None:
    if name in _LANG_CACHE:
        return _LANG_CACHE[name]
    try:
        from tree_sitter import Language
        if name == "python":
            import tree_sitter_python as g
            _LANG_CACHE[name] = Language(g.language())
        elif name == "javascript":
            import tree_sitter_javascript as g
            _LANG_CACHE[name] = Language(g.language())
        elif name == "typescript":
            import tree_sitter_typescript as g
            _LANG_CACHE[name] = Language(g.language_typescript())
        elif name == "tsx":
            import tree_sitter_typescript as g
            _LANG_CACHE[name] = Language(g.language_tsx())
        else:
            return None
    except (ImportError, AttributeError, Exception):
        return None
    return _LANG_CACHE[name]


EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
}


# ---------------------------------------------------------------------------
# Parsed result
# ---------------------------------------------------------------------------

@dataclass
class ParsedFile:
    path: str
    language: str
    symbols: list[SymbolNode] = field(default_factory=list)
    imports: list[dict[str, Any]] = field(default_factory=list)  # {type, module, name?, alias?}
    exports: list[str] = field(default_factory=list)
    call_refs: list[CallRef] = field(default_factory=list)
    raw_source: bytes = field(default=b"", repr=False)
    tree: Any = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text(node: Any, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _make_id(file_path: str, qualified_name: str) -> str:
    return f"{file_path}::{qualified_name}"


# ---------------------------------------------------------------------------
# Python extractor
# ---------------------------------------------------------------------------

def _extract_python(root_node: Any, src: bytes, fpath: str) -> tuple[
    list[SymbolNode], list[dict], list[str], list[CallRef]
]:
    symbols: list[SymbolNode] = []
    imports: list[dict] = []
    exports: list[str] = []
    call_refs: list[CallRef] = []

    def visit(node: Any, class_ctx: str | None = None, func_ctx: str | None = None):
        t = node.type

        if t == "function_definition":
            name_node = node.child_by_field_name("name")
            if not name_node:
                for child in node.children:
                    visit(child, class_ctx, func_ctx)
                return
            fname = _text(name_node, src)
            qualified = f"{class_ctx}.{fname}" if class_ctx else fname
            ntype = NodeType.METHOD if class_ctx else NodeType.FUNCTION
            # Extract first-line signature
            params_node = node.child_by_field_name("parameters")
            sig = f"def {qualified}{_text(params_node, src) if params_node else '()'}"
            sym = SymbolNode(
                id=_make_id(fpath, qualified),
                name=qualified,
                node_type=ntype,
                file_path=fpath,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=sig,
                language="python",
            )
            symbols.append(sym)
            # Recurse into body; nested functions are tracked with new func_ctx
            for child in node.children:
                visit(child, class_ctx, qualified)
            return

        if t == "class_definition":
            name_node = node.child_by_field_name("name")
            if not name_node:
                for child in node.children:
                    visit(child, class_ctx, func_ctx)
                return
            cname = _text(name_node, src)
            symbols.append(SymbolNode(
                id=_make_id(fpath, cname),
                name=cname,
                node_type=NodeType.CLASS,
                file_path=fpath,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                language="python",
            ))
            for child in node.children:
                visit(child, cname, func_ctx)
            return

        if t == "import_statement":
            for child in node.children:
                if child.type == "dotted_name":
                    imports.append({"type": "import", "module": _text(child, src), "alias": None})
                elif child.type == "aliased_import":
                    n = child.child_by_field_name("name")
                    a = child.child_by_field_name("alias")
                    if n:
                        imports.append({
                            "type": "import",
                            "module": _text(n, src),
                            "alias": _text(a, src) if a else None,
                        })

        elif t == "import_from_statement":
            mod_node = node.child_by_field_name("module_name")
            module = _text(mod_node, src) if mod_node else ""
            for child in node.children:
                if child.type in ("dotted_name", "identifier") and child != mod_node:
                    imports.append({"type": "from", "module": module, "name": _text(child, src)})

        # Capture function calls within a function context for CALLS edges
        elif t == "call" and func_ctx:
            func_node = node.child_by_field_name("function")
            if func_node:
                raw_callee = _text(func_node, src)
                # Normalise: strip self/cls receiver ("self.foo" → "foo")
                if "." in raw_callee:
                    parts = raw_callee.split(".")
                    if parts[0] in ("self", "cls", "this"):
                        raw_callee = ".".join(parts[1:])
                call_refs.append(CallRef(
                    caller_id=_make_id(fpath, func_ctx),
                    callee_name=raw_callee,
                    line=node.start_point[0] + 1,
                ))

        for child in node.children:
            visit(child, class_ctx, func_ctx)

    visit(root_node)
    return symbols, imports, exports, call_refs


# ---------------------------------------------------------------------------
# JavaScript / TypeScript extractor
# ---------------------------------------------------------------------------

def _extract_js_ts(root_node: Any, src: bytes, fpath: str, lang: str) -> tuple[
    list[SymbolNode], list[dict], list[str], list[CallRef]
]:
    symbols: list[SymbolNode] = []
    imports: list[dict] = []
    exports: list[str] = []
    call_refs: list[CallRef] = []

    def visit(node: Any, class_ctx: str | None = None, func_ctx: str | None = None):
        t = node.type

        # Named function declaration
        if t == "function_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                fname = _text(name_node, src)
                qualified = f"{class_ctx}.{fname}" if class_ctx else fname
                symbols.append(SymbolNode(
                    id=_make_id(fpath, qualified),
                    name=qualified,
                    node_type=NodeType.METHOD if class_ctx else NodeType.FUNCTION,
                    file_path=fpath,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    language=lang,
                ))
                for child in node.children:
                    visit(child, class_ctx, qualified)
                return

        # Arrow / function expression assigned to variable
        elif t in ("lexical_declaration", "variable_declaration"):
            for child in node.children:
                if child.type == "variable_declarator":
                    vname = child.child_by_field_name("name")
                    val = child.child_by_field_name("value")
                    if vname and val and val.type in ("arrow_function", "function"):
                        fname = _text(vname, src)
                        symbols.append(SymbolNode(
                            id=_make_id(fpath, fname),
                            name=fname,
                            node_type=NodeType.FUNCTION,
                            file_path=fpath,
                            line_start=node.start_point[0] + 1,
                            line_end=node.end_point[0] + 1,
                            language=lang,
                        ))
                        for sub in val.children:
                            visit(sub, class_ctx, fname)
                        continue
                    visit(child, class_ctx, func_ctx)
            return

        # Class declaration / expression
        elif t in ("class_declaration", "class"):
            name_node = node.child_by_field_name("name")
            cname = _text(name_node, src) if name_node else f"<anonymous@{node.start_point[0]+1}>"
            symbols.append(SymbolNode(
                id=_make_id(fpath, cname),
                name=cname,
                node_type=NodeType.CLASS,
                file_path=fpath,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                language=lang,
            ))
            for child in node.children:
                visit(child, cname, func_ctx)
            return

        # Method inside a class
        elif t == "method_definition":
            name_node = node.child_by_field_name("name")
            if name_node and class_ctx:
                mname = _text(name_node, src)
                qualified = f"{class_ctx}.{mname}"
                symbols.append(SymbolNode(
                    id=_make_id(fpath, qualified),
                    name=qualified,
                    node_type=NodeType.METHOD,
                    file_path=fpath,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    language=lang,
                ))
                for child in node.children:
                    visit(child, class_ctx, qualified)
                return

        # ESM import
        elif t == "import_statement":
            src_node = node.child_by_field_name("source")
            if src_node:
                module = _text(src_node, src).strip("'\"")
                imports.append({"type": "esm", "module": module})

        # CommonJS require
        elif t == "call_expression":
            fn = node.child_by_field_name("function")
            args = node.child_by_field_name("arguments")
            if fn and _text(fn, src) == "require" and args:
                for child in args.children:
                    if child.type == "string":
                        module = _text(child, src).strip("'\"")
                        imports.append({"type": "cjs", "module": module})
            # Also capture calls for CALLS edges
            if func_ctx and fn:
                raw_callee = _text(fn, src)
                if "." in raw_callee:
                    parts = raw_callee.split(".")
                    if parts[0] in ("this", "self"):
                        raw_callee = ".".join(parts[1:])
                call_refs.append(CallRef(
                    caller_id=_make_id(fpath, func_ctx),
                    callee_name=raw_callee,
                    line=node.start_point[0] + 1,
                ))

        # TypeScript interface
        elif t == "interface_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                iname = _text(name_node, src)
                symbols.append(SymbolNode(
                    id=_make_id(fpath, iname),
                    name=iname,
                    node_type=NodeType.INTERFACE,
                    file_path=fpath,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    language=lang,
                ))

        # Export — recurse into the declaration
        elif t in ("export_statement", "export_default_declaration"):
            decl = node.child_by_field_name("declaration")
            if decl:
                visit(decl, class_ctx, func_ctx)
                return

        for child in node.children:
            visit(child, class_ctx, func_ctx)

    visit(root_node)
    return symbols, imports, exports, call_refs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_file(file_path: str) -> ParsedFile | None:
    """Parse a source file and return its symbols, imports, and call references."""
    path = Path(file_path)
    ext = path.suffix.lower()
    lang = EXT_TO_LANG.get(ext)
    if not lang:
        return None

    language = _get_language(lang)
    if not language:
        return None

    try:
        source = path.read_bytes()
    except OSError:
        return None

    try:
        from tree_sitter import Parser
        parser = Parser(language)
    except TypeError:
        # tree-sitter < 0.22 compatibility
        from tree_sitter import Parser
        parser = Parser()
        parser.set_language(language)  # type: ignore[attr-defined]

    tree = parser.parse(source)
    pf = ParsedFile(path=file_path, language=lang, raw_source=source, tree=tree)

    if lang == "python":
        pf.symbols, pf.imports, pf.exports, pf.call_refs = _extract_python(
            tree.root_node, source, file_path
        )
    elif lang in ("javascript", "typescript", "tsx"):
        pf.symbols, pf.imports, pf.exports, pf.call_refs = _extract_js_ts(
            tree.root_node, source, file_path, lang
        )

    return pf
