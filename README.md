# Pocket Agent

> **Build a fully functional local coding agent from scratch.**
> 15 chapters · one notebook · no API key needed.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alankarmisra/pocket-agent/blob/main/pocket_agent_colab.ipynb)

---

Most agent tutorials hand you a framework and hide the hard parts.
This one builds every piece by hand — context budgets, file retrieval, planning, reflection — so you understand what the frameworks are actually doing.

By the end you have a working coding agent that can read a codebase, plan multi-step tasks, write and diff files, run tests, and revise its own plan when something goes wrong.

## Run it

**Google Colab (recommended) — no install needed**

Click the badge above. Sign in with a Google account. Run cells top to bottom.
Uses the free Gemini model via Colab AI — no API key, no quota setup.

**Local Ollama**

```bash
ollama pull qwen3-coder-next:cloud   # or any model you prefer
ollama pull nomic-embed-text          # for Chapter 9 semantic search
```

Open the notebook, run the Ollama setup cells in Chapter 1 instead of the Colab ones. Everything else is identical.

## What you build, chapter by chapter

| Ch | Title | What you gain |
|----|-------|---------------|
| 1  | Setup | Configure the notebook for Colab or Ollama |
| 2  | Hello LLM | Talk to an LLM from Python |
| 3  | Context is a Budget | Understand why context management exists |
| 4  | Give it a Map | Load AGENTS.md as an advisory project manifest |
| 5  | Glob + JIT Reads | Navigate a file tree without bulk-loading files |
| 6  | Fuzzy Scoring | Rank retrieved files, not just find them |
| 7  | Grep | Find code by content, not just filename |
| 8  | Microcompaction | Hot/cold storage — survive long sessions |
| 9  | Semantic RAG | Retrieval that understands meaning, not just keywords |
| 10 | Full Pipeline | One `run(query, repo)` call does everything |
| 11 | Write + Diff | The agent modifies files on disk |
| 12 | Agent Loop | Autonomous read → plan → write → verify loop |
| 13 | Test Generation | Agent writes and verifies its own tests |
| 14 | Adding a Capability | The three-step pattern for extending the agent |
| 15 | Reflection | ReAct loop — observe each step, revise the plan |

Each chapter is self-contained. Run through Chapter 10 and you have a RAG pipeline. Stop at Chapter 12 and you have an autonomous agent loop. The notebook is the source of truth — there is no separate library to install.

## Key functions you'll write

| Function | What it does |
|---|---|
| `chat()` | Single gateway to the LLM — every other function calls this |
| `glob_files()` + `jit_read()` | Navigate a repo without loading everything |
| `score_file()` + `semantic_retrieve()` | Rank files by relevance |
| `compact()` | Evict cold context to survive long sessions |
| `run()` | Full retrieval pipeline in one call |
| `write_file()` + `make_diff()` | Propose and preview file changes |
| `agent_loop()` | Plan → execute → verify |
| `agent_loop_reflect()` | Plan → execute → reflect → revise → repeat |

## What's inside the notebook

- A **sample project** (`sample_project/`) used as a practice codebase throughout
- A working **AGENTS.md** the agent uses as its project map
- Collapsible output panels so long agent runs don't flood the screen
- Every intermediate result visible but folded — click to expand

## After you finish

Point the agent at a real project:

```python
log = agent_loop_reflect(
    task      = "Add type hints to all functions in src/",
    repo_root = "/path/to/your/project",
)
```

The agent reads your codebase, plans the changes, writes diffs, and verifies the result — all from a single task string.
