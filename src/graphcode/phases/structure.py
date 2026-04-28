"""Phase 1: Walk the file tree and map folder/file relationships."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pathspec

_IGNORE_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    "env", ".env", "dist", "build", ".next", ".nuxt", ".turbo", "coverage",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", "eggs",
}

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".js", ".mjs", ".cjs", ".jsx",
    ".ts", ".tsx",
})


@dataclass
class FileNode:
    path: str       # absolute path
    rel_path: str   # relative to scan root
    name: str
    extension: str
    size: int
    parent_dir: str


@dataclass
class StructureMap:
    root: str
    files: list[FileNode] = field(default_factory=list)
    dirs: list[str] = field(default_factory=list)
    file_count: int = 0
    dir_count: int = 0

    def files_by_extension(self) -> dict[str, list[FileNode]]:
        result: dict[str, list[FileNode]] = {}
        for f in self.files:
            result.setdefault(f.extension, []).append(f)
        return result

    def files_in_dir(self, rel_dir: str) -> list[FileNode]:
        return [f for f in self.files if f.parent_dir == rel_dir]


def _load_gitignore(root: Path) -> pathspec.PathSpec | None:
    gi = root / ".gitignore"
    if gi.exists():
        return pathspec.PathSpec.from_lines("gitwildmatch", gi.read_text().splitlines())
    return None


def walk_tree(root: str, extensions: frozenset[str] | None = None) -> StructureMap:
    """
    Recursively walk *root*, collecting source files.
    Respects .gitignore and skips common noise directories.
    """
    exts = extensions if extensions is not None else SUPPORTED_EXTENSIONS
    root_path = Path(root).resolve()
    spec = _load_gitignore(root_path)
    sm = StructureMap(root=str(root_path))

    for dirpath, dirnames, filenames in os.walk(root_path, topdown=True):
        current = Path(dirpath)
        rel_current = current.relative_to(root_path)

        # Prune unwanted directories in-place so os.walk won't descend into them
        dirnames[:] = [
            d for d in sorted(dirnames)
            if d not in _IGNORE_DIRS
            and not d.startswith(".")
            and not (spec and spec.match_file(str(rel_current / d)))
        ]

        sm.dirs.append(str(rel_current))
        sm.dir_count += 1

        for fname in sorted(filenames):
            fpath = current / fname
            ext = fpath.suffix.lower()
            if ext not in exts:
                continue
            rel = str(fpath.relative_to(root_path))
            if spec and spec.match_file(rel):
                continue
            try:
                size = fpath.stat().st_size
            except OSError:
                continue
            sm.files.append(FileNode(
                path=str(fpath),
                rel_path=rel,
                name=fname,
                extension=ext,
                size=size,
                parent_dir=str(rel_current),
            ))
            sm.file_count += 1

    return sm
