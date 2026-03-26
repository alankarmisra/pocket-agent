"""
Pocket Agent — core pipeline
=============================
Every function built across Chapters 2–13, assembled in order.
Constants at the top; swap model or paths and everything adjusts.

New in this file (not in the notebook):
  chat_stream() — streaming token generator for the Streamlit UI
  retrieve()    — retrieval without generation, used by the Streamlit UI
"""

# ── Standard library ──────────────────────────────────────────────────────────
import difflib
import fnmatch
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import requests
from rich.console import Console
from rich.panel   import Panel
from rich.syntax  import Syntax
from rich.table   import Table
from rich.text    import Text

def fetch_url(url: str, max_chars: int = 8000) -> dict:
    """
    Fetch *url* and return plain text (HTML tags stripped).
    Returns {"url": str, "text": str, "ok": bool, "error": str | None}
    max_chars caps the text fed to the model — full pages can be huge.
    """
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "pocket-agent/1.0"})
        resp.raise_for_status()
        text = re.sub(r"<[^>]+>", " ", resp.text)
        text = re.sub(r"\s{2,}", "\n", text).strip()
        return {"url": url, "text": text[:max_chars], "ok": True, "error": None}
    except Exception as exc:
        return {"url": url, "text": "", "ok": False, "error": str(exc)}


# ══════════════════════════════════════════════════════════════════════════════
# Constants — edit these to point at a different model or project
# ══════════════════════════════════════════════════════════════════════════════

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "qwen3-coder-next:cloud"  # chat model
OLLAMA_EMBED    = "nomic-embed-text"         # embedding model (Chapter 9)
REPO_ROOT       = "."                        # default project directory
TOKEN_BUDGET    = 262144                     # context-window size in tokens
REBUILD_INDEX   = False

# Files the agent will read and embed — extend as needed
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".go", ".rs", ".java",
    ".c", ".cpp", ".h", ".md", ".txt", ".yaml", ".toml", ".json",
}

# Directories to skip when walking the repo
SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".ipynb_checkpoints", ".mypy_cache", ".pytest_cache",
}

# Budget thresholds (fraction of TOKEN_BUDGET)
HOT_THRESHOLD      = 0.70   # stop loading files once prompt exceeds 70 % of budget
COMPACT_THRESHOLD  = 0.80   # trigger compaction at 80 %
EVICT_TARGET       = 0.55   # compact down to 55 %

MANIFEST_FILENAME  = "AGENTS.md"

# ══════════════════════════════════════════════════════════════════════════════
# Chapter 2 — token counting and Ollama connectivity
# ══════════════════════════════════════════════════════════════════════════════

def count_tokens(text: str) -> int:
    """
    Estimate token count using the 4-chars-per-token heuristic.
    Accurate to ~15 % for English prose and source code.
    """
    return max(1, len(text) // 4)


def ping_ollama() -> tuple[bool, list[str] | str]:
    """
    Return (ok, info).
    ok=True  → info is a list of pulled model names.
    ok=False → info is the error message.
    """
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        return True, models
    except Exception as exc:
        return False, str(exc)


def chat(
    messages: list[dict],
    model: str = OLLAMA_MODEL,
) -> tuple[str, int]:
    """
    Send *messages* to Ollama, return (reply_text, tokens_used).
    tokens_used is the sum of prompt and completion tokens as reported by Ollama.
    """
    payload = {
        "model":   model,
        "messages": messages,
        "stream":  False,
        "options": {"num_ctx": TOKEN_BUDGET},
    }
    r = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=120)
    r.raise_for_status()
    data   = r.json()
    reply  = data["message"]["content"]
    tokens = data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
    return reply, tokens


def chat_stream(
    messages: list[dict],
    model: str = OLLAMA_MODEL,
):
    """
    Like chat() but yields tokens one at a time as Ollama produces them.
    Used by the Streamlit app to stream responses into the UI.

    Usage:
        for token in chat_stream(messages):
            print(token, end="", flush=True)

    Or with Streamlit:
        st.write_stream(chat_stream(messages))
    """
    payload = {
        "model":   model,
        "messages": messages,
        "stream":  True,
        "options": {"num_ctx": TOKEN_BUDGET},
    }
    with requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        stream=True,
        timeout=120,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                data = json.loads(line)
                if not data.get("done", False):
                    yield data["message"]["content"]


# ══════════════════════════════════════════════════════════════════════════════
# Chapter 2 — status panel (rich terminal output, used in notebook)
# ══════════════════════════════════════════════════════════════════════════════

console = Console()


def _budget_bar(used: int, total: int, width: int = 32) -> Text:
    """Render a coloured progress bar for the token budget."""
    pct    = used / total if total > 0 else 0
    filled = int(width * pct)
    color  = "green" if pct < 0.6 else ("yellow" if pct < 0.85 else "red")
    bar    = Text()
    bar.append("█" * filled,          style=color)
    bar.append("░" * (width - filled), style="dim")
    bar.append(f"  {used:,} / {total:,} tokens")
    return bar


def show_panel(
    query:        str,
    token_used:   int,
    token_budget: int        = TOKEN_BUDGET,
    files:        list[dict] | None = None,
    strategy:     str        = "none",
    prompt_size:  int        = 0,
) -> None:
    """Print the Pocket Agent status panel to the terminal."""
    files = files or []
    grid  = Table.grid(padding=(0, 2))
    grid.add_column(style="bold cyan", no_wrap=True, min_width=14)
    grid.add_column()
    grid.add_row("Token Budget", _budget_bar(token_used, token_budget))
    grid.add_row("", "")
    if files:
        lines = Text()
        for f in files:
            badge = (
                Text(" HOT  ", style="bold white on red")
                if f["hot"]
                else Text(" COLD ", style="white on blue")
            )
            lines.append_text(badge)
            lines.append(f"  {f['path']}\n")
        grid.add_row("Retrieved", lines)
    else:
        grid.add_row("Retrieved", Text("(none)", style="dim"))
    grid.add_row("Strategy",    Text(strategy,                  style="bold magenta"))
    grid.add_row("Prompt size", Text(f"{prompt_size:,} tokens", style="dim"))
    console.print(
        Panel(
            grid,
            title=(
                f"[bold]Pocket Agent[/bold]  ·  "
                f"[italic]{query[:70]}{'…' if len(query) > 70 else ''}[/italic]"
            ),
            border_style="bright_blue",
            expand=False,
        )
    )


# ══════════════════════════════════════════════════════════════════════════════
# Chapter 4 — manifest
# ══════════════════════════════════════════════════════════════════════════════

def load_manifest(repo_root: str = REPO_ROOT) -> dict:
    """
    Read AGENTS.md from *repo_root*.
    Returns {"text", "tokens", "paths", "found"}.
    """
    manifest_path = Path(repo_root) / MANIFEST_FILENAME
    if not manifest_path.exists():
        return {"text": "", "tokens": 0, "paths": [], "found": False}
    text  = manifest_path.read_text(errors="ignore")
    paths = re.findall(r'\b[\w./\-]+\.[\w]+\b', text)
    seen, unique_paths = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)
    return {"text": text, "tokens": count_tokens(text), "paths": unique_paths, "found": True}


def ask_with_manifest(
    query:     str,
    repo_root: str = REPO_ROOT,
    files:     list[dict] | None = None,
) -> tuple[str, int]:
    """
    Send *query* to Ollama with the project manifest as system prompt
    and any loaded files appended to the user message.
    Returns (reply_text, tokens_used).
    """
    manifest = load_manifest(repo_root)
    files    = files or []
    if manifest["found"]:
        system_content = (
            "You are a coding assistant with access to the project map below.\n"
            "Use it to understand the codebase structure before answering.\n\n"
            f"--- PROJECT MAP (AGENTS.md) ---\n{manifest['text']}\n---"
        )
    else:
        system_content = "You are a coding assistant."
    file_block = ""
    for f in files:
        file_block += f"\n\n--- FILE: {f['path']} ---\n{f.get('content', '')}"
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": query + file_block},
    ]
    reply, tokens_used = chat(messages)
    return reply, tokens_used


# ══════════════════════════════════════════════════════════════════════════════
# Chapter 5 — glob + JIT reads
# ══════════════════════════════════════════════════════════════════════════════

def glob_files(pattern: str, repo_root: str = REPO_ROOT) -> list[dict]:
    """
    Walk *repo_root* and return every file whose name matches *pattern*.
    Returns list of {"path", "bytes", "hot"} — no content loaded.
    """
    matches = []
    root    = Path(repo_root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for fname in filenames:
            if fnmatch.fnmatch(fname, pattern):
                full = Path(dirpath) / fname
                try:
                    size = full.stat().st_size
                except OSError:
                    continue
                matches.append({"path": str(full.relative_to(root)), "bytes": size, "hot": False})
    return sorted(matches, key=lambda x: x["path"])


def jit_read(file_meta: dict, repo_root: str = REPO_ROOT) -> dict:
    """
    Read the content of one file. Returns the same dict with
    "content", "tokens", and hot=True added.
    """
    full_path = Path(repo_root) / file_meta["path"]
    try:
        content = full_path.read_text(errors="ignore")
    except OSError as exc:
        content = f"[error reading file: {exc}]"
    return {**file_meta, "content": content, "tokens": count_tokens(content), "hot": True}


def budget_load(
    candidates:   list[dict],
    already_used: int   = 0,
    repo_root:    str   = REPO_ROOT,
    threshold:    float = HOT_THRESHOLD,
) -> list[dict]:
    """
    JIT-read candidates until the next file would push usage past
    threshold × TOKEN_BUDGET. Returns full list with hot/cold flags set.
    """
    budget_limit = int(TOKEN_BUDGET * threshold)
    used         = already_used
    result       = []
    for meta in candidates:
        headroom  = budget_limit - used
        estimated = meta["bytes"] // 4          # cheap byte-count proxy for token size
        if estimated > headroom:
            result.append({**meta, "hot": False})
            continue
        loaded = jit_read(meta, repo_root=repo_root)
        if used + loaded["tokens"] > budget_limit:
            result.append({**meta, "hot": False})
        else:
            used += loaded["tokens"]
            result.append(loaded)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Chapter 6 — fuzzy scoring
# ══════════════════════════════════════════════════════════════════════════════

_STOP_WORDS = {
    "a", "an", "the", "is", "in", "it", "of", "to", "do", "does",
    "how", "what", "why", "when", "where", "which", "who", "for",
    "and", "or", "but", "not", "with", "this", "that", "are", "was",
    "i", "me", "my", "we", "our", "you", "your",
}


def tokenize_query(query: str) -> list[str]:
    """Lowercase, split, remove stop words and tokens shorter than 3 chars."""
    words = re.split(r"[^a-zA-Z0-9]+", query.lower())
    return [w for w in words if len(w) >= 3 and w not in _STOP_WORDS]


def score_file(meta: dict, query_terms: list[str]) -> float:
    """Score *meta* against *query_terms* using path-only fuzzy matching."""
    if not query_terms:
        return 1.0
    path  = meta["path"].lower()
    parts = Path(path).parts
    stem  = Path(path).stem
    dirs  = parts[:-1]
    score = 1.0
    for term in query_terms:
        if term == stem:
            score += 4.0
        elif term in stem or stem in term:
            score += 2.5
        else:
            score += difflib.SequenceMatcher(None, term, stem).ratio() * 2.0
        for d in dirs:
            if term == d:
                score += 2.0
            elif term in d:
                score += 1.5
            else:
                score += difflib.SequenceMatcher(None, term, d).ratio() * 1.3
    if "test" in path and "test" not in query_terms:
        score *= 1.5
    return round(score, 4)


def rank_files(candidates: list[dict], query: str) -> list[dict]:
    """Return *candidates* sorted by relevance to *query*, highest first."""
    terms  = tokenize_query(query)
    scored = [{**c, "score": score_file(c, terms)} for c in candidates]
    return sorted(scored, key=lambda x: x["score"], reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# Chapter 7 — grep retrieval
# ══════════════════════════════════════════════════════════════════════════════

def grep_repo(
    pattern:       str,
    repo_root:     str = REPO_ROOT,
    extensions:    set = CODE_EXTENSIONS,
    context_lines: int = 2,
    max_matches:   int = 5,
) -> list[dict]:
    """
    Regex-search every source file under *repo_root*.
    Returns one dict per file that had at least one match, sorted by hit count.
    """
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error:
        regex = re.compile(re.escape(pattern), re.IGNORECASE)
    results = []
    root    = Path(repo_root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for fname in filenames:
            full = Path(dirpath) / fname
            if full.suffix.lower() not in extensions:
                continue
            try:
                lines = full.read_text(errors="ignore").splitlines()
            except OSError:
                continue
            hit_indices = [i for i, ln in enumerate(lines) if regex.search(ln)]
            if not hit_indices:
                continue
            excerpts, seen_lines = [], set()
            for hit_i in hit_indices[:max_matches]:
                start = max(0, hit_i - context_lines)
                end   = min(len(lines), hit_i + context_lines + 1)
                block = []
                for ln_i in range(start, end):
                    if ln_i not in seen_lines:
                        prefix = "→ " if ln_i == hit_i else "  "
                        block.append(f"{prefix}{ln_i+1:4d} │ {lines[ln_i]}")
                        seen_lines.add(ln_i)
                excerpts.append("\n".join(block))
            results.append({
                "path":      str(full.relative_to(root)),
                "hit_count": len(hit_indices),
                "excerpt":   "\n\n".join(excerpts),
                "bytes":     full.stat().st_size,
                "hot":       False,
                "score":     1.0,
            })
    return sorted(results, key=lambda x: x["hit_count"], reverse=True)


def query_to_patterns(query: str) -> str:
    """Convert a query into a regex pattern for grep_repo()."""
    terms = [t for t in tokenize_query(query) if len(t) >= 4]
    if not terms:
        return ""
    return "|".join(re.escape(t) for t in terms)


def grep_rank(query: str, repo_root: str = REPO_ROOT) -> list[dict]:
    """
    Full grep-based retrieval: derive pattern → grep → combine
    grep hit count and fuzzy path score → sort combined score descending.
    """
    pattern = query_to_patterns(query)
    if not pattern:
        return []
    hits = grep_repo(pattern, repo_root=repo_root)
    if not hits:
        return []
    terms          = tokenize_query(query)
    max_hits       = max(h["hit_count"] for h in hits) or 1
    max_path_score = max(score_file(h, terms) for h in hits) or 1
    ranked = []
    for h in hits:
        norm_grep = h["hit_count"]        / max_hits
        norm_path = score_file(h, terms)  / max_path_score
        ranked.append({**h, "score": round(norm_grep + norm_path, 4)})
    return sorted(ranked, key=lambda x: x["score"], reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# Chapter 8 — microcompaction
# ══════════════════════════════════════════════════════════════════════════════

def eviction_candidates(
    files:  list[dict],
    query:  str,
    n_keep: int = 1,
) -> tuple[list[dict], list[dict]]:
    """
    Split *files* into (to_evict, to_keep).
    Evicts the lowest-scoring HOT files, keeping *n_keep* HOT files.
    COLD files are never touched.
    """
    hot  = [f for f in files if f.get("hot")]
    cold = [f for f in files if not f.get("hot")]
    if len(hot) <= n_keep:
        return [], files
    terms  = tokenize_query(query)
    scored = sorted(hot, key=lambda f: f.get("score", score_file(f, terms)))
    return scored[:-n_keep], scored[-n_keep:] + cold


def summarise_file(file_dict: dict) -> dict:
    """
    Ask the LLM to compress *file_dict["content"]* to a 3–5 sentence summary.
    Returns a new dict with content replaced, tokens updated, hot=False.
    """
    content = file_dict.get("content", "")
    if not content.strip():
        return {**file_dict, "hot": False, "summary": True}
    prompt = (
        f"Summarise the following source file in 3–5 sentences of plain prose. "
        f"Name the key functions/classes and what they do. "
        f"No code blocks, no bullet points.\n\n"
        f"FILE: {file_dict['path']}\n\n{content}"
    )
    summary_text, _ = chat([{"role": "user", "content": prompt}])
    return {
        **file_dict,
        "content": f"[SUMMARY of {file_dict['path']}]\n{summary_text}",
        "tokens":  count_tokens(summary_text),
        "hot":     False,
        "summary": True,
    }


def compact(
    files:      list[dict],
    query:      str,
    token_used: int,
) -> tuple[list[dict], int, list[str]]:
    """
    If *token_used* exceeds COMPACT_THRESHOLD × TOKEN_BUDGET, summarise the
    least-relevant HOT files until usage drops below EVICT_TARGET × TOKEN_BUDGET.
    Returns (updated_files, new_token_used, log_lines).
    """
    threshold = int(TOKEN_BUDGET * COMPACT_THRESHOLD)
    target    = int(TOKEN_BUDGET * EVICT_TARGET)
    if token_used <= threshold:
        return files, token_used, []
    log     = [f"Compaction triggered: {token_used:,} / {TOKEN_BUDGET:,} tokens"]
    updated = list(files)
    used    = token_used
    while used > target:
        to_evict, _ = eviction_candidates(updated, query, n_keep=1)
        if not to_evict:
            break
        victim     = to_evict[0]
        before     = victim.get("tokens", 0)
        summarised = summarise_file(victim)
        after      = summarised["tokens"]
        updated    = [summarised if f["path"] == victim["path"] else f for f in updated]
        used      -= before - after
        log.append(f"  compacted {victim['path']}: {before}t → {after}t")
    log.append(f"After compaction: {used:,} / {TOKEN_BUDGET:,} tokens")
    return updated, used, log


# ══════════════════════════════════════════════════════════════════════════════
# Chapter 9 — semantic RAG
# ══════════════════════════════════════════════════════════════════════════════

# In-memory cache: repo_root → list of embedded file dicts
_EMBED_INDEX: dict[str, list[dict]] = {}


def embed(text: str, model: str = OLLAMA_EMBED) -> np.ndarray:
    """
    Return a unit-normalised embedding vector for *text*.
    Uses Ollama's /api/embed endpoint; falls back to /api/embeddings for
    older Ollama installations.
    """
    payload = {"model": model, "input": text}
    try:
        r = requests.post(f"{OLLAMA_BASE_URL}/api/embed", json=payload, timeout=60)
        r.raise_for_status()
        vec = r.json()["embeddings"][0]
    except Exception:
        payload_old = {"model": model, "prompt": text}
        r = requests.post(f"{OLLAMA_BASE_URL}/api/embeddings", json=payload_old, timeout=60)
        r.raise_for_status()
        vec = r.json()["embedding"]
    arr  = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 0 else arr


def build_index(
    repo_root:  str  = REPO_ROOT,
    extensions: set  = CODE_EXTENSIONS,
    force:      bool = REBUILD_INDEX,
) -> list[dict]:
    """
    Embed every source file under *repo_root* and cache the result in memory.
    Subsequent calls return the cache unless force=True.
    """
    if repo_root in _EMBED_INDEX and not force:
        return _EMBED_INDEX[repo_root]
    files = glob_files("*", repo_root=repo_root)
    files = [f for f in files if Path(f["path"]).suffix.lower() in extensions]
    index = []
    for f in files:
        full_path = Path(repo_root) / f["path"]
        try:
            content = full_path.read_text(errors="ignore")
        except OSError:
            continue
        if not content.strip():
            continue                  # skip empty files — embed() returns shape (0,)
        vec = embed(content[:4000])   # truncate very long files before embedding
        if len(vec) == 0:
            continue                  # embedding model returned nothing
        index.append({**f, "vector": vec, "content": content, "tokens": count_tokens(content)})
    _EMBED_INDEX[repo_root] = index
    return index


def semantic_retrieve(
    query:     str,
    repo_root: str = REPO_ROOT,
    top_k:     int = 5,
) -> list[dict]:
    """
    Return the top-k files ranked by cosine similarity to *query*.
    Calls build_index() — cached after the first call per repo_root.
    """
    index     = build_index(repo_root=repo_root)
    query_vec = embed(query)
    if len(query_vec) == 0:
        return []   # query embedding failed — fall back gracefully
    scored    = []
    for entry in index:
        vec = entry["vector"]
        if len(vec) == 0 or vec.shape != query_vec.shape:
            continue  # stale/bad vector — skip rather than crash
        sim = float(np.dot(query_vec, vec))
        scored.append({
            "path":    entry["path"],
            "bytes":   entry["bytes"],
            "tokens":  entry["tokens"],
            "content": entry["content"],
            "hot":     False,
            "score":   round(sim, 4),
        })
    return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
# Chapter 10 — full pipeline
# ══════════════════════════════════════════════════════════════════════════════

_CONCEPTUAL_WORDS = {
    "why", "how", "explain", "what", "design", "architecture",
    "pattern", "approach", "concept", "idea", "strategy", "logic",
    "purpose", "reason", "difference", "relationship",
}

_GREP_SIGNALS = re.compile(
    r'(?:"[^"]+"'
    r'|`[^`]+`'
    r'|raise\s+\w+'
    r'|def\s+\w+'
    r'|class\s+\w+'
    r'|import\s+\w+'
    r'|\b[A-Z][a-zA-Z]+Error\b)'
)


def pick_strategy(query: str) -> str:
    """
    Classify *query* into "grep", "semantic", or "fuzzy".
    grep     — looking for a specific symbol or string
    semantic — conceptual / meaning-based question
    fuzzy    — everything else
    """
    if _GREP_SIGNALS.search(query):
        return "grep"
    if set(query.lower().split()) & _CONCEPTUAL_WORDS:
        return "semantic"
    return "fuzzy"


class RunResult(NamedTuple):
    reply:       str
    strategy:    str
    files:       list[dict]
    tokens_used: int
    compact_log: list[str]


def run(
    query:     str,
    repo_root: str = REPO_ROOT,
    strategy:  str = "auto",
    top_k:     int = 8,
) -> RunResult:
    """
    Full retrieval + generation pipeline.
    strategy: "auto" | "fuzzy" | "grep" | "semantic"
    """
    manifest     = load_manifest(repo_root)
    manifest_tok = manifest["tokens"]
    strat        = pick_strategy(query) if strategy == "auto" else strategy

    if strat == "grep":
        candidates  = grep_rank(query, repo_root=repo_root)
        found_paths = {f["path"] for f in candidates}
        extras      = [f for f in rank_files(glob_files("*", repo_root=repo_root), query)
                       if f["path"] not in found_paths]
        candidates  = (candidates + extras)[:top_k]
    elif strat == "semantic":
        candidates = semantic_retrieve(query, repo_root=repo_root, top_k=top_k)
    else:
        candidates = rank_files(glob_files("*", repo_root=repo_root), query)[:top_k]

    loaded   = budget_load(candidates, already_used=manifest_tok, repo_root=repo_root)
    hot_tok  = sum(f.get("tokens", 0) for f in loaded if f["hot"])
    total    = manifest_tok + hot_tok + count_tokens(query)
    loaded, total, compact_log = compact(loaded, query, total)
    hot_files = [f for f in loaded if f["hot"]]
    reply, tokens_used = ask_with_manifest(query, repo_root=repo_root, files=hot_files)

    return RunResult(
        reply=reply, strategy=strat, files=loaded,
        tokens_used=tokens_used, compact_log=compact_log,
    )


def retrieve(
    query:     str,
    repo_root: str = REPO_ROOT,
    strategy:  str = "auto",
    top_k:     int = 8,
) -> tuple[list[dict], list[dict], str, int, list[str]]:
    """
    Run the full retrieval pipeline WITHOUT generating a reply.
    Returns (messages, loaded_files, strategy_used, total_tokens, compact_log).

    The returned *messages* list is ready to pass to chat() or chat_stream().
    Used by the Streamlit app so it can stream the response itself.
    """
    manifest     = load_manifest(repo_root)
    manifest_tok = manifest["tokens"]
    strat        = pick_strategy(query) if strategy == "auto" else strategy

    if strat == "grep":
        candidates  = grep_rank(query, repo_root=repo_root)
        found_paths = {f["path"] for f in candidates}
        extras      = [f for f in rank_files(glob_files("*", repo_root=repo_root), query)
                       if f["path"] not in found_paths]
        candidates  = (candidates + extras)[:top_k]
    elif strat == "semantic":
        candidates = semantic_retrieve(query, repo_root=repo_root, top_k=top_k)
    else:
        candidates = rank_files(glob_files("*", repo_root=repo_root), query)[:top_k]

    loaded   = budget_load(candidates, already_used=manifest_tok, repo_root=repo_root)
    hot_tok  = sum(f.get("tokens", 0) for f in loaded if f["hot"])
    total    = manifest_tok + hot_tok + count_tokens(query)
    loaded, total, compact_log = compact(loaded, query, total)
    hot_files = [f for f in loaded if f["hot"]]

    # Build the prompt messages the same way ask_with_manifest() does
    if manifest["found"]:
        system = (
            "You are a coding assistant with access to the project map below.\n"
            f"--- PROJECT MAP ---\n{manifest['text']}\n---"
        )
    else:
        system = "You are a coding assistant."

    file_block = ""
    for f in hot_files:
        file_block += f"\n\n--- FILE: {f['path']} ---\n{f.get('content', '')}"

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": query + file_block},
    ]
    return messages, loaded, strat, total, compact_log


# ══════════════════════════════════════════════════════════════════════════════
# Chapter 11 — write + diff
# ══════════════════════════════════════════════════════════════════════════════

def make_diff(original: str, proposed: str, file_path: str = "<file>", context: int = 3) -> str:
    """Return a unified diff string between *original* and *proposed*."""
    orig_lines = original.splitlines(keepends=True)
    new_lines  = proposed.splitlines(keepends=True)
    diff_lines = list(difflib.unified_diff(
        orig_lines, new_lines,
        fromfile=f"a/{file_path}", tofile=f"b/{file_path}", n=context,
    ))
    return "".join(diff_lines)


def apply_patch(
    file_path:   str,
    instruction: str,
    repo_root:   str = REPO_ROOT,
) -> tuple[str, str]:
    """
    Ask the LLM to apply *instruction* to *file_path*.
    Returns (original_content, proposed_content).
    If the file does not exist, original is empty — allows creating new files.
    """
    # Strip repo_root prefix if the model accidentally included it
    rel = file_path
    for prefix in (repo_root + "/", repo_root + os.sep):
        if rel.startswith(prefix):
            rel = rel[len(prefix):]
            break

    full_path = Path(repo_root) / rel
    original  = full_path.read_text(errors="ignore") if full_path.exists() else ""

    prompt = (
        f"Apply the following instruction to the source file below.\n"
        f"Return ONLY the complete modified file — no explanations, "
        f"no markdown fences, no commentary.\n\n"
        f"INSTRUCTION: {instruction}\n\n"
        f"FILE: {rel}\n```\n{original}\n```"
    )
    proposed_raw, _ = chat([{"role": "user", "content": prompt}])
    proposed = re.sub(r"^```[a-zA-Z]*\n?", "", proposed_raw.strip())
    proposed = re.sub(r"\n?```$", "", proposed)
    return original, proposed.strip() + "\n"


def write_file(
    file_path:   str,
    instruction: str,
    repo_root:   str  = REPO_ROOT,
    dry_run:     bool = True,
) -> tuple[str, str, str]:
    """
    Apply *instruction* to *file_path*, print the diff, optionally write to disk.
    Returns (original, proposed, diff_text).
    """
    original, proposed = apply_patch(file_path, instruction, repo_root)
    diff = make_diff(original, proposed, file_path)
    if not dry_run and diff:
        full_path = Path(repo_root) / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(proposed)
    return original, proposed, diff


# ══════════════════════════════════════════════════════════════════════════════
# Chapter 12 — agent loop
# ══════════════════════════════════════════════════════════════════════════════

_PLAN_SYSTEM = """\
You are a coding agent. Given a task and a project map, produce an ordered
list of ALL steps needed to fully complete the task.

Each step must be one of these four formats (blank line between steps):

STEP: <one-sentence description>
ACTION: read
TARGET: <question to answer about the codebase>

STEP: <one-sentence description>
ACTION: write
TARGET: <file path relative to repo root>
INSTRUCTION: <precise, self-contained instruction for what to write>

STEP: <one-sentence description>
ACTION: bash
CMD: <shell command, e.g. cp foo.md bar.md>

STEP: <one-sentence description>
ACTION: fetch
URL: <full URL to retrieve>

Rules:
- Prefer ACTION: bash for file system operations (copy, move, delete, rename).
- Use ACTION: fetch when the task references an external URL or documentation link.
- Use ACTION: read only when you need code understanding before you can write.
- Use ACTION: write when the LLM must generate or edit file content.
- Emit at most 6 steps. Do not add any text outside the step blocks.
"""


def plan_task(task: str, repo_root: str = REPO_ROOT) -> list[dict]:
    """
    Ask the LLM to produce a step-by-step plan for *task*.
    Returns a list of {"step", "action", "target", "instruction", "cmd", "url"} dicts.
    """
    manifest = load_manifest(repo_root)
    prompt   = f"PROJECT MAP:\n{manifest['text']}\n\nTASK: {task}"
    raw, _   = chat([
        {"role": "system", "content": _PLAN_SYSTEM},
        {"role": "user",   "content": prompt},
    ])
    steps = []
    for block in re.split(r"\n{2,}", raw.strip()):
        lines = {
            m.group(1).lower(): m.group(2).strip()
            for line in block.splitlines()
            if (m := re.match(
                r"^(STEP|ACTION|TARGET|INSTRUCTION|CMD|URL):\s*(.+)$",
                line, re.IGNORECASE
            ))
        }
        action = lines.get("action", "").lower()
        has_content = (
            "target" in lines
            or "cmd"  in lines
            or "url"  in lines
            or action in ("bash", "fetch")
        )
        if action and has_content:
            steps.append({
                "step":        lines.get("step", ""),
                "action":      action,
                "target":      lines.get("target", ""),
                "instruction": lines.get("instruction", ""),
                "cmd":         lines.get("cmd", ""),
                "url":         lines.get("url", ""),
            })
    return steps

def execute_step(
    step:      dict,
    repo_root: str  = REPO_ROOT,
) -> dict:
    """
    Execute one plan step.  Returns a structured result dict:

      read  → {"type": "read",  "output": str}
      write → {"type": "write", "file_path": str, "new_content": str, "diff": str}
              The file is NOT written to disk — the caller decides whether to apply it.
      bash  → {"type": "bash",  "cmd": str, "output": str, "ok": bool}
      fetch → {"type": "fetch", "url": str, "output": str, "ok": bool}
    """
    action = step.get("action", "read")

    if action == "fetch":
        url    = step.get("url", step.get("target", ""))
        result = fetch_url(url)
        if not result["ok"]:
            return {"type": "fetch", "url": url, "output": f"Error: {result['error']}", "ok": False}
        summary, _ = chat([
            {"role": "system", "content": "Summarise the following web page concisely."},
            {"role": "user",   "content": f"URL: {url}\n\n{result['text']}"},
        ])
        return {"type": "fetch", "url": url, "output": summary, "ok": True}

    if action == "bash":
        cmd = step.get("cmd", "")
        proc = subprocess.run(
            cmd, shell=True, cwd=repo_root,
            capture_output=True, text=True,
        )
        output = (proc.stdout + proc.stderr).strip()
        return {"type": "bash", "cmd": cmd, "output": output, "ok": proc.returncode == 0}

    if action == "write":
        original, proposed, diff = write_file(
            file_path=step["target"], instruction=step["instruction"],
            repo_root=repo_root, dry_run=True,
        )
        return {
            "type":        "write",
            "file_path":   step["target"],
            "new_content": proposed,
            "diff":        diff,
        }

    # read (default)
    result = run(step["target"], repo_root=repo_root)
    return {"type": "read", "output": result.reply}

def agent_loop(
    task:      str,
    repo_root: str  = REPO_ROOT,
    dry_run:   bool = True,
    max_steps: int  = 8,
) -> list[dict]:
    """
    Autonomous task execution loop: plan → execute → verify.
    Returns the step log (list of dicts with "step", "action", "status").
    """
    plan = plan_task(task, repo_root=repo_root)
    if not plan:
        return []
    log = []
    for step in plan[:max_steps]:
        status = execute_step(step, repo_root=repo_root, dry_run=dry_run)
        log.append({**step, "status": status})
    # Verify: ask the model to review what was changed
    written = [s for s in log if s["action"] == "write"]
    if written:
        verify_query = (
            f"Review whether this task has been completed correctly: {task}\n"
            f"Files modified: {', '.join(s['target'] for s in written)}"
        )
        run(verify_query, repo_root=repo_root)
    return log


# ══════════════════════════════════════════════════════════════════════════════
# Chapter 13 — test generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_tests(source_path: str, repo_root: str = REPO_ROOT) -> str:
    """
    Generate a pytest test file for the module at *source_path*.
    Returns the test file content as a string (not written to disk yet).
    """
    source = (Path(repo_root) / source_path).read_text(errors="ignore")
    prompt = (
        f"Write a complete pytest test file for the following Python module.\n\n"
        f"Requirements:\n"
        f"- Use pytest (not unittest)\n"
        f"- Cover the main happy path and at least two edge cases per function\n"
        f"- Import from the module using its dotted path relative to the repo root\n"
        f"- Return ONLY the test file — no explanation, no markdown fences\n\n"
        f"SOURCE FILE: {source_path}\n\n{source}"
    )
    raw, _ = chat([{"role": "user", "content": prompt}])
    code   = re.sub(r"^```[a-zA-Z]*\n?", "", raw.strip())
    code   = re.sub(r"\n?```$", "", code)
    return code.strip() + "\n"


def run_tests(test_path: str, repo_root: str = REPO_ROOT) -> dict:
    """
    Run pytest on *test_path* (relative to repo_root).
    Returns {"passed", "failed", "errors", "output", "ok"}.
    """
    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_path, "-v", "--tb=short", "--no-header", "-q"],
        capture_output=True, text=True, cwd=repo_root,
    )
    output = result.stdout + result.stderr
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", output)) else 0
    failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", output)) else 0
    errors = int(m.group(1)) if (m := re.search(r"(\d+) error",  output)) else 0
    return {"passed": passed, "failed": failed, "errors": errors, "output": output,
            "ok": passed > 0 and failed == 0 and errors == 0}


def test_loop(
    source_path: str,
    repo_root:   str = REPO_ROOT,
    max_retries: int = 3,
) -> dict:
    """
    Full generate → run → fix loop.
    Returns the final run_tests() result dict plus an "attempts" key.
    """
    stem      = Path(source_path).stem
    test_path = f"tests/test_{stem}_gen.py"
    full_test = Path(repo_root) / test_path
    full_test.parent.mkdir(parents=True, exist_ok=True)

    test_code = generate_tests(source_path, repo_root=repo_root)
    full_test.write_text(test_code)

    for attempt in range(1, max_retries + 2):
        result = run_tests(test_path, repo_root=repo_root)
        if result["ok"]:
            result["attempts"] = attempt
            return result
        if attempt > max_retries:
            break
        fix_instruction = (
            f"The test file has failures. Fix ONLY the test code.\n"
            f"Pytest output:\n\n{result['output'][-1500:]}"
        )
        _, fixed_code = apply_patch(test_path, fix_instruction, repo_root=repo_root)
        full_test.write_text(fixed_code)

    result["attempts"] = max_retries + 1
    return result
