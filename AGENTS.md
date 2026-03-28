# AGENTS.md — Pocket Agent project map

## What this repo is
A Jupyter notebook tutorial that builds a local coding agent step by step.
Each chapter adds one capability. The notebook IS the source of truth.

## Entry point
- `pocket_agent_colab.ipynb` — the entire project lives here

## Chapter guide
| Chapter | Topic | Key concepts defined |
|---------|-------|----------------------|
| 1 | Setup | `chat()`, `show_panel()`, `ping_ollama()` |
| 2 | Hello LLM | `count_tokens()`, `scan_repo_costs()` |
| 3 | Manifest | `load_manifest()`, `ask_with_manifest()` |
| 4 | Glob + JIT reads | `glob_files()`, `jit_read()` |
| 5 | Fuzzy scoring | `score_files()` |
| 6 | Grep | `grep_repo()` |
| 7 | Microcompaction | `compact()`, hot/cold store |
| 8 | Semantic RAG | `embed()`, `retrieve()` |
| 9 | Full pipeline | `run()` |
| 10 | Write + diff | `write_file()`, `make_diff()` |
| 11 | Agent loop | `agent_loop()` |
| 12 | Test generation | `generate_tests()` |

## Off-limits (never load these)
- `.git/` — git internals
- `__pycache__/` — compiled bytecode
- `*.ipynb_checkpoints/` — Jupyter autosave noise

## Questions about token budgeting → Chapter 3
## Questions about retrieval strategy → Chapters 4–8
## Questions about the agent loop → Chapter 11
