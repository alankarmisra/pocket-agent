"""
Pocket Agent — Streamlit UI
============================
Run with:  streamlit run agent/app.py
"""

import sys, os, time
from pathlib import Path

import streamlit as st

# Allow importing core.py from the same directory
sys.path.insert(0, str(Path(__file__).parent))
from core import (
    OLLAMA_MODEL, OLLAMA_EMBED, OLLAMA_BASE_URL,
    TOKEN_BUDGET, REPO_ROOT,
    retrieve, chat_stream,
    plan_task, execute_step,
    load_manifest,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pocket Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal CSS — dark panel for diffs, monospace chat ─────────────────────────
st.markdown("""
<style>
/* make code blocks a little tighter */
code { font-size: 0.82rem; }
/* diff panel */
.diff-box {
    background: #0d1117;
    color: #e6edf3;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 12px 16px;
    font-family: monospace;
    font-size: 0.78rem;
    white-space: pre;
    overflow-x: auto;
    max-height: 420px;
    overflow-y: auto;
}
.diff-box .add  { color: #3fb950; }
.diff-box .del  { color: #f85149; }
.diff-box .meta { color: #8b949e; }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ─────────────────────────────────────────────────────
def _init_state():
    defaults = {
        # shared
        "repo_root": REPO_ROOT,
        # ask tab
        "ask_history": [],          # list of {"role", "content", "files", "strategy"}
        # agent tab
        "agent_state": "idle",      # idle | planning | executing | waiting_confirm | done
        "agent_task":  "",
        "agent_plan":  [],
        "agent_step":  0,
        "agent_log":   [],          # list of strings
        "pending_patch": None,      # {"file_path", "new_content", "diff"}
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 Pocket Agent")
    st.caption("Local coding agent powered by Ollama")

    st.divider()
    st.subheader("Project")
    new_root = st.text_input(
        "Root path",
        value=st.session_state.repo_root,
        help="Absolute path to the repository you want to query or modify",
    )
    if new_root != st.session_state.repo_root:
        st.session_state.repo_root   = new_root
        st.session_state.ask_history = []   # clear history when project changes
        st.session_state.agent_state = "idle"
        st.rerun()

    st.divider()
    st.subheader("Model info")
    st.markdown(f"""
| | |
|---|---|
| **LLM** | `{OLLAMA_MODEL}` |
| **Embed** | `{OLLAMA_EMBED}` |
| **Context** | `{TOKEN_BUDGET:,}` tokens |
| **Base URL** | `{OLLAMA_BASE_URL}` |
""")

    # Show files that were loaded in the last Ask query
    if st.session_state.ask_history:
        last = st.session_state.ask_history[-1]
        if last.get("files"):
            st.divider()
            st.subheader("Context files")
            st.caption(f"Strategy: **{last.get('strategy','—')}**")
            for f in last["files"]:
                st.code(f, language=None)

    st.divider()
    if st.button("🗑️  Clear session", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        _init_state()
        st.rerun()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_ask, tab_agent = st.tabs(["💬  Ask", "🛠️  Agent"])

# ══════════════════════════════════════════════════════════════════════════════
# ASK TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_ask:
    st.header("Ask about your codebase")
    st.caption(
        "Ask any question — the agent retrieves relevant files, "
        "then streams the answer directly from the model."
    )

    # Render chat history
    for msg in st.session_state.ask_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("files"):
                with st.expander(
                    f"📂 {len(msg['files'])} file(s) read  ·  strategy: {msg.get('strategy','—')}",
                    expanded=False,
                ):
                    for f in msg["files"]:
                        st.code(f, language=None)

    # Chat input at the bottom
    if query := st.chat_input("Ask anything about your code…"):
        repo = st.session_state.repo_root

        # Show user bubble immediately
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.ask_history.append({"role": "user", "content": query})

        # Retrieve context (no LLM call yet)
        with st.spinner("Retrieving context…"):
            try:
                messages, loaded_files, strategy, total_tokens, compact_log = retrieve(
                    query, repo_root=repo
                )
            except Exception as e:
                st.error(f"Retrieval error: {e}")
                st.stop()

        # Stream the answer
        with st.chat_message("assistant"):
            files_placeholder = st.empty()
            if loaded_files:
                files_placeholder.caption(
                    f"📂 Reading {len(loaded_files)} file(s) · strategy: **{strategy}** · "
                    f"{total_tokens:,} tokens"
                )

            # st.write_stream() consumes the generator token by token
            try:
                full_reply = st.write_stream(chat_stream(messages))
            except Exception as e:
                st.error(f"LLM error: {e}")
                full_reply = ""

            # Show files expander beneath the response
            if loaded_files:
                with st.expander(
                    f"📂 {len(loaded_files)} file(s) read  ·  strategy: {strategy}",
                    expanded=False,
                ):
                    for f in loaded_files:
                        st.code(f, language=None)

        # Persist to history
        st.session_state.ask_history.append({
            "role":     "assistant",
            "content":  full_reply,
            "files":    loaded_files,
            "strategy": strategy,
        })

# ══════════════════════════════════════════════════════════════════════════════
# AGENT TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_agent:
    st.header("Agent — plan & execute")
    st.caption(
        "Give the agent a task. It will plan the steps, execute them one by one, "
        "and show you a diff before writing any file."
    )

    repo = st.session_state.repo_root

    # ── State: idle — show task input ──────────────────────────────────────────
    if st.session_state.agent_state == "idle":
        task = st.text_area(
            "Task",
            placeholder="e.g. Add a --verbose flag to main.py that prints each step",
            height=120,
        )
        if st.button("▶  Run agent", type="primary", disabled=not task.strip()):
            st.session_state.agent_task  = task.strip()
            st.session_state.agent_state = "planning"
            st.session_state.agent_log   = []
            st.session_state.agent_step  = 0
            st.session_state.agent_plan  = []
            st.rerun()

    # ── State: planning — call plan_task ───────────────────────────────────────
    elif st.session_state.agent_state == "planning":
        with st.spinner("Planning…"):
            try:
                plan = plan_task(st.session_state.agent_task, repo_root=repo)
                st.session_state.agent_plan  = plan
                st.session_state.agent_state = "executing"
                st.session_state.agent_log.append(
                    f"**Plan ({len(plan)} step{'s' if len(plan)!=1 else ''}):**"
                )
                for i, step in enumerate(plan, 1):
                    action_label = step.get("action", "?").upper()
                    step_desc    = step.get("step", str(step))
                    st.session_state.agent_log.append(f"  {i}. **{action_label}** — {step_desc}")
            except Exception as e:
                st.session_state.agent_log.append(f"❌ Planning failed: {e}")
                st.session_state.agent_state = "idle"
        st.rerun()

    # ── State: executing — run the next step ───────────────────────────────────
    elif st.session_state.agent_state == "executing":
        plan = st.session_state.agent_plan
        idx  = st.session_state.agent_step

        if idx >= len(plan):
            st.session_state.agent_state = "done"
            st.rerun()
        else:
            step      = plan[idx]
            step_desc = step.get("step", str(step))
            st.session_state.agent_log.append(f"\n⚙️  **Step {idx+1}/{len(plan)}:** {step_desc}")

            with st.spinner(f"Step {idx+1}/{len(plan)}: {step_desc}"):
                try:
                    result = execute_step(step, repo_root=repo)
                    if result.get("type") == "write":
                        # Pause and ask user to confirm before touching disk
                        st.session_state.pending_patch = {
                            "file_path":   result["file_path"],
                            "new_content": result["new_content"],
                            "diff":        result["diff"],
                        }
                        st.session_state.agent_state = "waiting_confirm"
                    elif result.get("type") == "bash":
                        # Shell command ran immediately — log the result
                        ok_icon = "✅" if result.get("ok") else "❌"
                        cmd_out = result.get("output", "")
                        st.session_state.agent_log.append(
                            f"   {ok_icon} `{result['cmd']}`"
                            + (f"\n   ```\n   {cmd_out[:300]}\n   ```" if cmd_out else "")
                        )
                        st.session_state.agent_step += 1
                    elif result.get("type") == "fetch":
                        # URL fetched and summarised — log the summary
                        ok_icon = "✅" if result.get("ok") else "❌"
                        out = result.get("output", "")
                        st.session_state.agent_log.append(
                            f"   {ok_icon} fetched `{result.get('url', '')}`"
                            + (f"\n\n   {out[:400]}{'…' if len(out) > 400 else ''}" if out else "")
                        )
                        st.session_state.agent_step += 1
                    else:
                        # read — log a preview and continue
                        output = result.get("output", "")
                        st.session_state.agent_log.append(
                            f"   ✅ {output[:300]}{'…' if len(output) > 300 else ''}"
                        )
                        st.session_state.agent_step += 1
                except Exception as e:
                    st.session_state.agent_log.append(f"   ❌ Error: {e}")
                    st.session_state.agent_step += 1
            st.rerun()

    # ── State: waiting_confirm — show diff, ask user ───────────────────────────
    elif st.session_state.agent_state == "waiting_confirm":
        patch = st.session_state.pending_patch

    # ── State: done ────────────────────────────────────────────────────────────
    elif st.session_state.agent_state == "done":
        pass  # just fall through to log display

    # ── Always render the activity log ────────────────────────────────────────
    if st.session_state.agent_log or st.session_state.agent_state != "idle":
        st.divider()
        st.subheader("Activity log")
        for line in st.session_state.agent_log:
            st.markdown(line)

    # ── Render diff confirm panel (outside the elif so it repaints correctly) ──
    if st.session_state.agent_state == "waiting_confirm":
        patch = st.session_state.pending_patch
        st.divider()
        st.subheader(f"📝 Proposed change to `{patch['file_path']}`")

        # Colour the diff lines
        diff_lines = patch["diff"].splitlines()
        coloured = []
        for ln in diff_lines:
            if ln.startswith("+") and not ln.startswith("+++"):
                coloured.append(f'<span class="add">{ln}</span>')
            elif ln.startswith("-") and not ln.startswith("---"):
                coloured.append(f'<span class="del">{ln}</span>')
            elif ln.startswith("@@"):
                coloured.append(f'<span class="meta">{ln}</span>')
            else:
                coloured.append(ln)
        html_diff = "<br>".join(coloured)
        st.markdown(f'<div class="diff-box">{html_diff}</div>', unsafe_allow_html=True)

        col_accept, col_skip, col_abort = st.columns([1, 1, 4])
        with col_accept:
            if st.button("✅  Accept", type="primary", use_container_width=True):
                try:
                    # Write the pre-generated content directly — no LLM call needed
                    full_path = Path(repo) / patch["file_path"]
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    full_path.write_text(patch["new_content"])
                    st.session_state.agent_log.append(
                        f"   ✅ Written: `{patch['file_path']}`"
                    )
                except Exception as e:
                    st.session_state.agent_log.append(f"   ❌ Write failed: {e}")
                st.session_state.pending_patch = None
                st.session_state.agent_step   += 1
                st.session_state.agent_state   = "executing"
                st.rerun()
        with col_skip:
            if st.button("⏭️  Skip", use_container_width=True):
                st.session_state.agent_log.append(
                    f"   ⏭️  Skipped: `{patch['file_path']}`"
                )
                st.session_state.pending_patch = None
                st.session_state.agent_step   += 1
                st.session_state.agent_state   = "executing"
                st.rerun()

    # ── Done banner ────────────────────────────────────────────────────────────
    if st.session_state.agent_state == "done":
        st.success("Agent finished all steps.")
        if st.button("Start new task"):
            st.session_state.agent_state = "idle"
            st.session_state.agent_log   = []
            st.session_state.agent_plan  = []
            st.session_state.agent_step  = 0
            st.rerun()
