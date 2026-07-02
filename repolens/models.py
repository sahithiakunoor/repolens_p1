from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class QueryIntent(Enum):
    EXPLAIN = "explain"    # "how does X work?"
    EXAMPLE = "example"    # "show me how to use X"
    FIND    = "find"       # "where is X defined?"
    DEBUG   = "debug"      # "why does X fail when..."
    COMPARE = "compare"    # "difference between X and Y?"


SUPPORTED_LANGUAGES = {
    ".py":   "python",
    ".js":   "javascript",
    ".ts":   "typescript",
    ".tsx":  "typescript",
    ".java": "java",
    ".go":   "go",
    ".rb":   "ruby",
    ".rs":   "rust",
    ".cpp":  "cpp",
    ".c":    "c",
}


@dataclass
class CodeChunk:
    chunk_id: str           # unique: filepath::name:start_line
    content: str            # raw source of the code unit
    chunk_type: str         # "function" | "class" | "method" | "module"
    name: str               # symbol name e.g. "RunnableSequence.invoke"
    file_path: str          # relative path inside the repo
    language: str           # "python", "javascript", etc.
    start_line: int
    end_line: int
    docstring: str          # extracted docstring / JSDoc
    parent_class: str       # class name if this is a method
    imports: list[str]      # top-level imports from the file
    repo_url: str           # base GitHub URL for building line links
    github_url: str         # direct link to exact line in GitHub


@dataclass
class RetrievedChunk:
    chunk: CodeChunk
    score: float            # final score after reranking


@dataclass
class RAGResponse:
    answer: str
    citations: list[RetrievedChunk]
    intent: QueryIntent
    latency_ms: int