# 🤖 Pocket Agent

> **Build a fully functional local coding agent from scratch.**
> 12 chapters. One Python file. No API costs.

Pocket Agent is a hands-on tutorial that takes you from a bare Ollama connection to a fully autonomous coding agent that reads, writes, and tests code — all running locally on your machine. No OpenAI account. No cloud bills. No black boxes.

Each chapter adds exactly one capability. Every chapter ends with a working, runnable script you can immediately test against your own codebase.

---

## What you'll build

By the end of the tutorial you'll have a coding agent that:

- Connects to a local LLM via [Ollama](https://ollama.com) (Mistral, CodeLlama, DeepSeek-Coder, or any compatible model)
- Navigates your codebase using four retrieval strategies borrowed from how Claude Code and Codex actually work
- Manages its own context window with a hot/cold microcompaction system so it doesn't go off the rails on long sessions
- Reads source files, proposes changes, and writes structured diffs back to disk
- Runs your test suite and iterates autonomously until tests pass
- Generates new tests for untested code

---

## Why build it from scratch?

There are already excellent agentic tools on the market. But understanding *how* they work — why they glob before they read, why they score filenames before loading them, why context compaction exists and what it costs you when it fires at the wrong moment — makes you dramatically better at using them and prompting them effectively.

This tutorial is the internals, made legible.

---

## Chapter overview

| Ch | Title | What you gain |
|----|-------|---------------|
| 1 | Hello Ollama | Talk to a local LLM from Python |
| 2 | Context is a Budget | Understand why context management exists |
| 3 | Give it a Map | Load AGENTS.md as an advisory project manifest |
| 4 | Glob + JIT Reads | Navigate a file tree without bulk-loading files |
| 5 | Fuzzy Scoring | Rank retrieved files, not just find them |
| 6 | Grep | Find code by content, not just filename |
| 7 | Microcompaction | Hot tail / cold storage — survive long sessions |
| 8 | Semantic RAG | Retrieval that understands meaning, not just keywords |
| 9 | Full Pipeline | One `run(query, repo)` call does everything |
| 9b | Web UI | Optional browser interface at localhost:8000 |
| 10 | Write + Diff | The agent modifies files on disk |
| 11 | Agent Loop | Autonomous read → plan → write → verify loop |
| 12 | Test Generation | Agent writes and verifies its own tests |

Chapters 1–9 live in a single evolving Python file. Chapters 10–12 split into separate modules as the concerns grow large enough to warrant it.

---

## Retrieval strategies

The pipeline blends two design philosophies — Codex's fuzzy filename scoring and Claude Code's glob-first JIT discipline — and layers semantic RAG on top of both.

| Strategy | Inspired by | What it does |
|----------|-------------|--------------|
| Glob + JIT reads | Claude Code | Directory listing first, targeted reads second. Never bulk-loads. |
| Fuzzy filename scoring | Codex | Scores files by prefix match, contiguous character runs, subsequence. |
| Grep / lexical search | Both | Regex search across file contents with surrounding context lines. |
| Semantic RAG | LlamaIndex | Tree-sitter AST chunking + vector index + Ollama embeddings. |

---

## Compaction: hot tail / cold storage

Introduced in Chapter 7. Modelled on Claude Code's microcompaction layer.

- **Hot tail** — the N most recent tool results stay fully inline in the prompt
- **Cold storage** — older results are offloaded, replaced by a path stub the agent can re-read if needed
- Chapter 7 deliberately fills the budget first, then shows compaction solving it — the problem before the fix

---

## Prerequisites

**Ollama** running locally with two models pulled:

```bash
ollama pull mistral           # or codellama, deepseek-coder, llama3
ollama pull nomic-embed-text  # embedding model, needed from Chapter 8
```

**Python dependencies** (introduced gradually — you don't install everything upfront):

```bash
pip install llama-index llama-index-llms-ollama llama-index-embeddings-ollama
pip install tree-sitter tree-sitter-python
pip install rich fastapi uvicorn   # rich from Ch 1, fastapi from Ch 9b
```

---

## Quickstart

```bash
git clone https://github.com/yourusername/pocket-agent
cd pocket-agent

# Point the agent at any local repo
export REPO_ROOT=/path/to/your/project
export OLLAMA_MODEL=mistral

# Start at Chapter 1
python agent.py
```

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `OLLAMA_MODEL` | `mistral` | LLM model name |
| `OLLAMA_EMBED` | `nomic-embed-text` | Embedding model for semantic search |
| `REPO_ROOT` | `.` | Repository to run the agent against |

---

## AGENTS.md

Before running the agent against your own project, drop an `AGENTS.md` in the repo root. The agent reads this as an advisory hint about your project structure, build commands, and conventions. Chapter 3 covers exactly what to put in it.

A minimal example:

```markdown
# My Project

A brief description of what this codebase does.

## Key Directories
- `src/`      - main source
- `tests/`    - test suite

## Build & Test
    make build
    pytest tests/

## Conventions
- Type hints on all functions
- Tests required for new features
```

---

## Background reading

This tutorial grew out of a deep dive into how Claude Code and Codex handle context management — specifically how they differ. The short version:

- **Codex** is explore-first, compact-if-needed. Fuzzy filename scoring, tool-driven exploration, single-threshold compaction.
- **Claude Code** is budget-first, explore-targeted. CLAUDE.md injected as system prompt, glob before read, three-layer compaction (micro → auto → delta summary).

Neither uses semantic RAG by default. Both treat the LLM's reasoning as the primary navigation tool, with retrieval as a targeted support layer rather than the main event. Pocket Agent takes the best of both and adds the RAG layer on top.

---

## Companion scripts

The repo also includes two standalone reference scripts from the tutorial design phase:

- `codex_context_prep.py` — models Codex's retrieval pipeline without LlamaIndex
- `claude_context_prep.py` — models Claude Code's pipeline including microcompaction

These are read-only, no-dependency scripts useful for understanding the concepts before the full LlamaIndex pipeline is introduced.

---

## License

MIT
