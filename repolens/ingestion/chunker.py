"""
AST-aware code chunker using tree-sitter.

For each file, we walk the AST and extract meaningful logical units:
  - functions / async functions
  - classes
  - methods (functions nested inside classes)

This is fundamentally better than token-based chunking because:
  - A function split across two chunks loses its meaning entirely
  - Two unrelated functions merged into one chunk pollute retrieval
  - Symbol names and docstrings are preserved as first-class metadata
"""

import uuid
from pathlib import Path
from typing import Optional

from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import tree_sitter_java as tsjava

from repolens.models import CodeChunk, SUPPORTED_LANGUAGES

# ── Language registry ────────────────────────────────────────────────────────
# We load grammars lazily so startup is fast even if a language isn't needed.

_GRAMMAR_LOADERS = {
    "python":     lambda: Language(tspython.language()),
    "java":       lambda: Language(tsjava.language()),
}

# Try optional languages — not all may be installed in the environment
try:
    import tree_sitter_javascript as tsjs
    _GRAMMAR_LOADERS["javascript"] = lambda: Language(tsjs.language())
except ImportError:
    pass

try:
    import tree_sitter_typescript as tsts
    _GRAMMAR_LOADERS["typescript"] = lambda: Language(tsts.language_typescript())
except ImportError:
    pass

_GRAMMAR_CACHE: dict[str, Language] = {}


def _get_grammar(language: str) -> Optional[Language]:
    if language not in _GRAMMAR_CACHE:
        loader = _GRAMMAR_LOADERS.get(language)
        if loader is None:
            return None
        _GRAMMAR_CACHE[language] = loader()
    return _GRAMMAR_CACHE[language]


# ── Node type maps per language ───────────────────────────────────────────────
# tree-sitter node type names differ slightly across grammars.

FUNCTION_NODES = {
    "python":     {"function_definition"},
    "javascript": {"function_declaration", "arrow_function", "method_definition"},
    "typescript": {"function_declaration", "arrow_function", "method_definition"},
    "java":       {"method_declaration", "constructor_declaration"},
}

CLASS_NODES = {
    "python":     {"class_definition"},
    "javascript": {"class_declaration"},
    "typescript": {"class_declaration"},
    "java":       {"class_declaration", "interface_declaration"},
}

DOCSTRING_NODES = {
    "python":     {"expression_statement"},   # first child is string literal
    "javascript": {"comment"},
    "typescript": {"comment"},
    "java":       {"block_comment"},
}


# ── Public API ────────────────────────────────────────────────────────────────

def detect_language(file_path: str) -> Optional[str]:
    """Return language string from file extension, or None if unsupported."""
    ext = Path(file_path).suffix.lower()
    return SUPPORTED_LANGUAGES.get(ext)


def chunk_file(
    source: str,
    file_path: str,
    repo_url: str,
    repo_root: str = "",
    default_branch: str = "main",
) -> list[CodeChunk]:
    """
    Parse a source file and return a list of CodeChunks.

    Args:
        source:         Raw file contents.
        file_path:      Absolute or repo-root-relative path to the file.
        repo_url:       Base GitHub URL (e.g. https://github.com/org/repo).
        repo_root:      Local path prefix to strip when building the relative
                        path used in GitHub URLs. E.g. '/tmp/clones/org__repo'.
        default_branch: The repo's default branch ('main', 'master', etc.).
                        Used to build accurate GitHub blob URLs.

    Falls back to whole-file chunking if:
      - The language is unsupported
      - tree-sitter fails to parse (syntax errors, etc.)
    """
    language = detect_language(file_path)
    if language is None:
        return []  # skip unsupported files silently

    grammar = _get_grammar(language)
    if grammar is None:
        # Language supported in principle but grammar not installed
        return _fallback_chunk(source, file_path, language, repo_url, repo_root, default_branch)

    try:
        return _ast_chunk(source, file_path, language, grammar, repo_url, repo_root, default_branch)
    except Exception:
        return _fallback_chunk(source, file_path, language, repo_url, repo_root, default_branch)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _relative_path(file_path: str, repo_root: str) -> str:
    """
    Strip the local repo_root prefix from file_path to get the path
    as it appears inside the GitHub repo (used in blob URLs and chunk IDs).
    """
    if repo_root and file_path.startswith(repo_root):
        rel = file_path[len(repo_root):]
        return rel.lstrip("/")
    return file_path


def _ast_chunk(
    source: str,
    file_path: str,
    language: str,
    grammar: Language,
    repo_url: str,
    repo_root: str,
    default_branch: str,
) -> list[CodeChunk]:
    parser = Parser(grammar)
    tree = parser.parse(bytes(source, "utf-8"))
    lines = source.splitlines()

    rel_path = _relative_path(file_path, repo_root)

    # Extract top-level imports once per file
    imports = _extract_imports(tree.root_node, lines, language)

    chunks: list[CodeChunk] = []
    fn_types  = FUNCTION_NODES.get(language, set())
    cls_types = CLASS_NODES.get(language, set())

    def visit(node, parent_class: str = ""):
        node_type = node.type

        if node_type in cls_types:
            name = _node_name(node, lines)
            start = node.start_point[0]
            end   = node.end_point[0]
            docstring = _extract_docstring(node, lines, language)

            # Find where the class's methods begin. We emit a compact "header"
            # chunk covering the class signature, docstring, and any class-level
            # attributes — but NOT the method bodies, which are chunked
            # separately below. This avoids duplicating the entire class body
            # into one giant, semantically-meaningless chunk (a 4000-line class
            # would otherwise embed to a useless averaged vector and get
            # truncated by the cross-encoder at rerank time).
            first_method_line = _first_child_def_line(node, fn_types)

            if first_method_line is not None:
                header_end = first_method_line - 1
                content = "\n".join(lines[start:header_end + 1])
                chunk_end_line = header_end + 1
            else:
                # No methods — a small class (e.g. a dataclass). Keep it whole.
                content = "\n".join(lines[start:end + 1])
                chunk_end_line = end + 1

            chunks.append(_make_chunk(
                content=content,
                chunk_type="class",
                name=name,
                rel_path=rel_path,
                language=language,
                start_line=start + 1,
                end_line=chunk_end_line,
                docstring=docstring,
                parent_class="",
                imports=imports,
                repo_url=repo_url,
                default_branch=default_branch,
            ))

            # Visit children so methods inside the class are also extracted
            for child in node.children:
                visit(child, parent_class=name)

        elif node_type in fn_types:
            name = _node_name(node, lines)
            if not name:
                return  # skip anonymous functions

            start = node.start_point[0]
            end   = node.end_point[0]
            content = "\n".join(lines[start:end + 1])
            docstring = _extract_docstring(node, lines, language)

            qualified_name = f"{parent_class}.{name}" if parent_class else name
            chunk_type = "method" if parent_class else "function"

            chunks.append(_make_chunk(
                content=content,
                chunk_type=chunk_type,
                name=qualified_name,
                rel_path=rel_path,
                language=language,
                start_line=start + 1,
                end_line=end + 1,
                docstring=docstring,
                parent_class=parent_class,
                imports=imports,
                repo_url=repo_url,
                default_branch=default_branch,
            ))

        else:
            # Keep walking for nested definitions
            for child in node.children:
                visit(child, parent_class=parent_class)

    for child in tree.root_node.children:
        visit(child)

    return chunks


def _make_chunk(
    content, chunk_type, name, rel_path, language,
    start_line, end_line, docstring, parent_class, imports,
    repo_url, default_branch,
) -> CodeChunk:
    """
    Build a CodeChunk. rel_path is the repo-relative path (no local prefix),
    used both for the chunk_id and the GitHub blob URL.
    """
    github_url = (
        f"{repo_url.rstrip('/')}/blob/{default_branch}/{rel_path}"
        f"#L{start_line}-L{end_line}"
    )
    return CodeChunk(
        chunk_id=f"{rel_path}::{name}:{start_line}",
        content=content,
        chunk_type=chunk_type,
        name=name,
        file_path=rel_path,
        language=language,
        start_line=start_line,
        end_line=end_line,
        docstring=docstring,
        parent_class=parent_class,
        imports=imports,
        repo_url=repo_url,
        github_url=github_url,
    )


def _first_child_def_line(class_node, fn_types: set) -> int | None:
    """
    Return the 0-indexed start line of the first method/function defined
    directly inside a class body, or None if the class has no methods.

    Used to bound the class "header" chunk so it stops before the method
    bodies (which are chunked separately). We search the class's block/body
    node for the earliest function-definition child.
    """
    # Find the body node (the indented block under the class)
    body = None
    for child in class_node.children:
        if child.type in {"block", "class_body", "declaration_list"}:
            body = child
            break
    if body is None:
        return None

    earliest = None
    for child in body.children:
        if child.type in fn_types:
            line = child.start_point[0]
            if earliest is None or line < earliest:
                earliest = line
        # Handle decorated methods: the decorator wraps the function node
        if child.type == "decorated_definition":
            for inner in child.children:
                if inner.type in fn_types:
                    line = child.start_point[0]  # include the decorator line
                    if earliest is None or line < earliest:
                        earliest = line
    return earliest


def _node_name(node, lines: list[str]) -> str:
    """Extract the identifier name from a function/class node."""
    for child in node.children:
        if child.type in {"identifier", "property_identifier", "type_identifier"}:
            start = child.start_point
            return lines[start[0]][start[1]:child.end_point[1]]
    return ""


def _extract_docstring(node, lines: list[str], language: str) -> str:
    """
    Extract the docstring or leading comment from a function/class node.
    For Python: first child that is an expression_statement containing a string.
    For others: leading block/line comment before the node.
    """
    if language == "python":
        body_node = None
        for child in node.children:
            if child.type == "block":
                body_node = child
                break
        if body_node and body_node.children:
            first = body_node.children[0]
            if first.type == "expression_statement" and first.children:
                inner = first.children[0]
                if inner.type == "string":
                    s = inner.start_point[0]
                    e = inner.end_point[0]
                    raw = "\n".join(lines[s:e + 1]).strip('"""').strip("'''").strip()
                    return raw
    return ""


def _extract_imports(root_node, lines: list[str], language: str) -> list[str]:
    """Collect top-level import statements from the file."""
    import_types = {
        "python":     {"import_statement", "import_from_statement"},
        "javascript": {"import_statement"},
        "typescript": {"import_statement"},
        "java":       {"import_declaration"},
    }
    target_types = import_types.get(language, set())
    imports = []
    for child in root_node.children:
        if child.type in target_types:
            s = child.start_point[0]
            e = child.end_point[0]
            imports.append("\n".join(lines[s:e + 1]))
    return imports


def _fallback_chunk(
    source: str,
    file_path: str,
    language: str,
    repo_url: str,
    repo_root: str,
    default_branch: str,
) -> list[CodeChunk]:
    """
    When AST parsing fails, treat the whole file as a single chunk.
    Better than silently dropping the file.
    """
    rel_path = _relative_path(file_path, repo_root)
    name = Path(rel_path).stem
    lines = source.splitlines()
    return [_make_chunk(
        content=source,
        chunk_type="module",
        name=name,
        rel_path=rel_path,
        language=language,
        start_line=1,
        end_line=len(lines),
        docstring="",
        parent_class="",
        imports=[],
        repo_url=repo_url,
        default_branch=default_branch,
    )]