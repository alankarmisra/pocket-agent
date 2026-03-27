# Pocket Agent

This is an MD export of pocket_agent.ipynb. Download and open the notebook to run the samples yourself. 

> **Build a fully functional local coding agent from scratch.**
> 16 chapters · one notebook · no API costs.

Each chapter adds exactly one capability.
Run a chapter's cells top-to-bottom and you get a working agent at the end of that chapter.

| Ch | Title | What you gain |
|----|-------|---------------|
| 1  | Setup | Install Ollama, pull a model, configure the notebook |
| 2  | Hello Ollama | Talk to a local LLM from Python |
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
| 15 | Streamlit App | A browser UI that uses everything you built |
| 16 | LlamaIndex | The same pipeline with a production-grade library |

---


---
## Chapter 1 — Setup

Get Ollama installed, pull a model, and configure the notebook.

| Step | Type | What it does |
|------|------|-------------|
| 1.1 | read | Install Ollama — follow the instructions |
| 1.2 | run  | Pull your chosen model |
| 1.3 | run  | Auto-configure token budget and settings |

### 1.1 Install Ollama

Download and install Ollama for your platform from **[ollama.com/download](https://ollama.com/download)**.

Once installed, start the Ollama daemon if it isn't already running:

```bash
ollama serve
```

> On macOS and Windows, Ollama starts automatically when you open the app.  
> On Linux, run `ollama serve` in a separate terminal and leave it open.

Browse available models at **[ollama.com/library](https://ollama.com/library)**.  
Cloud models (tagged `cloud`) run on Ollama's servers — no GPU or download required, just `ollama signin` first.

### 1.2 Pull a model

Set `MODEL_TO_PULL` below to whichever model you want, then run the cell.  
It calls `ollama pull` for you — no terminal needed. We use `qwen3-coder-next` 
for demonstration purposes, but you can use any model (preferably a coding model). 

**Recommendations:**

| Model | Size |
|-------|------|
| `qwen3-coder-next:cloud` | - (cloud) |
| `qwen3-coder-next` | ~52 GB (local) | 

For embeddings (Chapter 9) `nomic-embed-text` is always pulled alongside your chat model.

**The nomic-embed-text model**

Unlike chat models, `nomic-embed-text` doesn't generate text — it converts a piece of text into a 
list of numbers (a vector) that represents its meaning. Two pieces of text that mean similar things 
will produce similar vectors. Chapter 9 uses this to find files that are *semantically* related to 
your query, even if they share no keywords in common. It's small (~274 MB) because it only needs to 
encode meaning, not generate language.


```python
import subprocess, sys

# ── Change this to the model you want ────────────────────────────────────────
MODEL_TO_PULL = "qwen3-coder-next:cloud"   # ← edit me

def _pull(model: str) -> None:
    """Pull an Ollama model, suppressing the spinner output that clutters notebooks."""
    print(f"Pulling {model} ...", end=" ", flush=True)
    result = subprocess.run(
        ["ollama", "pull", model],
        stdout=subprocess.DEVNULL,   # progress bars go to stdout
        stderr=subprocess.DEVNULL,   # spinner frames go to stderr
    )
    if result.returncode == 0:
        print("done ✓")
    else:
        print("FAILED ✗")
        raise RuntimeError(f"ollama pull {model} exited with code {result.returncode}")

_pull(MODEL_TO_PULL)
_pull("nomic-embed-text")   # needed for Chapter 9 semantic search

print("\nAll models ready. Proceed to cell 1.3.")

```

    Pulling qwen3-coder-next:cloud ... done ✓
    Pulling nomic-embed-text ... done ✓
    
    All models ready. Proceed to cell 1.3.


### 1.3 Run configuration

Sets up all the variables the notebook uses throughout every chapter. The token budget is 
auto-detected from the model you pulled in 1.2 — you don't need to change anything here 
unless you want to point the agent at a different folder (`REPO_ROOT`) or swap models later.

## Global Configuration


```python
# ── Known context windows — add your model here if it's missing ──────────────
import os

_CONTEXT_WINDOWS = {
    "qwen3-coder-next": 262144,
    "qwen3-coder":      131072,
    "qwen4.5":           32768,
    "qwen3":             32768,
    "llama4.2":          32768,
    "llama4.1":          32768,
    "mistral":           32768,
    "codellama":          4096,
    "gemma3":            32768,
    "devstral-small-2":  32768,
}

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = MODEL_TO_PULL                    # set in cell 1.2
OLLAMA_EMBED    = "nomic-embed-text"
REPO_ROOT       = "."
USE_WEB_UI      = False
REBUILD_INDEX   = False

# Auto-detect token budget from model name prefix
_base        = OLLAMA_MODEL.split(":")[0]          # strip tag  e.g. "qwen4.5:4b" → "qwen4.5"
if _base not in _CONTEXT_WINDOWS:
    print(f"WARNING: '{_base}' not in _CONTEXT_WINDOWS — defaulting to 32,768 tokens.")
    print(f"  Add it to _CONTEXT_WINDOWS above if you know the correct context length.")
    print(f"  Run: ollama show {OLLAMA_MODEL}  and look for 'context length'")
TOKEN_BUDGET = _CONTEXT_WINDOWS.get(_base, 32768)  # default 32K if model not in table

# Cap num_ctx to what the running Ollama server actually accepts;
# override with OLLAMA_NUM_CTX env var if needed.
SAFE_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", str(TOKEN_BUDGET)))

print(f"Model        : {OLLAMA_MODEL}")
print(f"Embed model  : {OLLAMA_EMBED}")
print(f"Token budget : {TOKEN_BUDGET:,}")
print(f"Repo root    : {REPO_ROOT}")
```

    Model        : qwen3-coder-next:cloud
    Embed model  : nomic-embed-text
    Token budget : 262,144
    Repo root    : .


---
## Chapter 2 — Hello Ollama

**Goal:** create the two things everything else builds on —
a function that talks to Ollama, and a status panel that exposes the
agent's internal state at every step.

**You will:**
- Verify Ollama is reachable
- Write `chat()` — the single function every later chapter calls
- Write `show_panel()` — an HTML panel showing the token budget, retrieved files,
  active strategy, and assembled prompt size — rendered natively in the notebook
- Run a test query end-to-end

> **Token budget** is how much of the model's context window your current prompt is using.
> Every model has a fixed limit (e.g. 32,768 tokens). As the agent loads more files into context,
> the budget fills up. When it's full, no more files can be added — the model simply won't see them.
> `show_panel()` makes this visible so you always know how much room you have left.


### 2.1 Dependencies

Chapter 2 only needs `requests` (HTTP to Ollama).
`IPython.display` is part of Jupyter itself — no install needed.
Later chapters will `pip install` their own additions at the top of their section.



```python
# Ensure Chapter 2 dependencies are installed in the current Python environment.
import subprocess, sys

_CH2_DEPS = ["requests"]   # requests: HTTP calls to Ollama
                            # IPython.display is bundled with Jupyter — no install needed

for _pkg in _CH2_DEPS:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", _pkg],
        stdout=subprocess.DEVNULL,
    )

print("Chapter 2 dependencies ready:", _CH2_DEPS)

```

    Chapter 2 dependencies ready: ['requests']


### 2.2 Checking Ollama is running

Before sending any messages we probe the `/api/tags` endpoint.
That endpoint returns the list of locally pulled models — a useful
sanity check that both the server and the model we want are present.


```python
import requests

def ping_ollama() -> tuple[bool, list[str] | str]:
    """
    Return (ok, info).
    ok=True  → info is a list of pulled model names.
    ok=False → info is the error message.
    """
    try:
        # GET /api/tags returns metadata for every model Ollama has pulled locally
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()                              # throw an error on 4xx / 5xx
        models = [m["name"] for m in r.json().get("models", [])]
        return True, models
    except Exception as exc:
        return False, str(exc)                            # return the error as-is

ok, info = ping_ollama()

if ok:
    print(f"Ollama is running.")
    print(f"Pulled models: {info}")

    # Soft warning — the rest of the notebook will fail if the model isn't present,
    # but we don't hard-crash here so the user can read the message clearly
    if OLLAMA_MODEL not in " ".join(info):
        print(f"\n  WARNING: '{OLLAMA_MODEL}' not found in pulled models.")
        print(f"  Run: ollama pull {OLLAMA_MODEL}")
else:
    print(f"Cannot reach Ollama at {OLLAMA_BASE_URL}")
    print(f"Error: {info}")
    print("Start it with: ollama serve")     # most common fix — Ollama daemon not running
```

    Ollama is running.
    Pulled models: ['nomic-embed-text:latest', 'qwen3-coder-next:cloud', 'qwen3.5:4b', 'deepseek-v3.1:671b-cloud', 'qwen3-coder:480b-cloud', 'codellama:7b']


### 2.3 The `chat()` function

`chat()` is the only function that ever calls Ollama.
Every chapter — from the simplest single-turn query to the full
autonomous agent loop — goes through this one function.

It takes a standard OpenAI-style `messages` list and returns
the reply text plus the token count Ollama reports.
The token count is what populates the budget bar in the status panel.


```python
def chat(
    messages: list[dict],
    model:    str = OLLAMA_MODEL,
) -> tuple[str, int]:
    """
    Send *messages* to Ollama, return (reply_text, tokens_used).

    If Ollama stops early (done_reason == "length") the reply is automatically
    continued until the model signals "stop".
    """
    full_reply   = ""
    total_tokens = 0
    msgs         = list(messages)
    while True:
        payload = {
            "model":    model,
            "messages": msgs,
            "stream":   False,
            "options":  {"num_ctx": min(TOKEN_BUDGET, SAFE_NUM_CTX)},
        }
        r = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=180)
        r.raise_for_status()
        data         = r.json()
        chunk        = data["message"]["content"]
        full_reply  += chunk
        total_tokens += data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
        if data.get("done_reason", "stop") != "length":
            break
        msgs = msgs + [
            {"role": "assistant", "content": chunk},
            {"role": "user",      "content": "Continue."},
        ]
    return full_reply, total_tokens
```

### 2.4 The status panel

The panel is the same in every chapter — only the data it receives changes.
It shows four things:

| Row | What it shows |
|-----|--------------|
| **Token Budget** | A coloured progress bar. Green < 60 %, yellow < 85 %, red above. |
| **Retrieved** | Every file the agent loaded, tagged HOT (in prompt) or COLD (offloaded). |
| **Strategy** | Which retrieval strategy was used: glob, fuzzy, grep, or semantic. |
| **Prompt size** | The assembled prompt in tokens, *before* the LLM call. |

`show_panel()` is intentionally stateless — you call it with the current
snapshot before the LLM call and again after, so you can see both.


```python
from IPython.display import HTML, Markdown, display


def show_rule(title: str = "") -> None:
    """Render a horizontal rule with a centred title — native notebook HTML."""
    safe = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    display(HTML(
        f'<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace">'
        f'<hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0">'
        f'<b style="color:#57606a;white-space:nowrap;font-size:0.9rem">{safe}</b>'
        f'<hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0">'
        f'</div>'
    ))


def show_panel(
    query:        str,
    token_used:   int,
    token_budget: int        = TOKEN_BUDGET,
    files:        list[dict] | None = None,
    strategy:     str        = "none",
    prompt_size:  int        = 0,
) -> None:
    """
    Render the Pocket Agent status panel as HTML in the notebook output.

    Parameters
    ----------
    query        : the user's query (shown in the panel title)
    token_used   : tokens consumed so far
    token_budget : total context-window size (default TOKEN_BUDGET)
    files        : list of dicts {"path": str, "hot": bool}
    strategy     : retrieval strategy name
    prompt_size  : estimated assembled-prompt size in tokens
    """
    files = files or []
    pct   = token_used / token_budget if token_budget > 0 else 0
    bar_w = min(int(pct * 200), 200)           # px, capped at full bar
    bar_color = (
        "#2da44e" if pct < 0.70 else
        "#bf8700" if pct < 0.85 else
        "#cf222e"
    )

    # ── file badges ────────────────────────────────────────────────────────────
    file_rows = ""
    for f in files:
        if f["hot"]:
            badge = '<span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>'
        else:
            badge = '<span style="background:#0969da;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">COLD</span>'
        path  = f["path"].replace("&","&amp;").replace("<","&lt;")
        file_rows += f'<div style="margin:2px 0">{badge}&nbsp;&nbsp;<code style="font-size:0.85rem">{path}</code></div>'

    if not file_rows:
        file_rows = '<span style="color:#8c959f;font-style:italic">none</span>'

    short_q = query[:80] + ("…" if len(query) > 80 else "")
    short_q = short_q.replace("&","&amp;").replace("<","&lt;")

    html = f"""
<div style="border:1px solid #d0d7de;border-radius:8px;padding:14px 18px;
            margin:8px 0;background:#f6f8fa;font-family:monospace;font-size:0.88rem">
  <div style="font-weight:bold;color:#0969da;margin-bottom:10px">
    Pocket Agent&nbsp;&nbsp;·&nbsp;&nbsp;<em style="font-weight:normal;color:#57606a">{short_q}</em>
  </div>
  <table style="border-collapse:collapse;width:100%">
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:middle;white-space:nowrap">Token Budget</td>
      <td style="vertical-align:middle">
        <div style="display:inline-flex;align-items:center;gap:10px">
          <div style="width:200px;height:10px;border-radius:5px;background:#e0e0e0;overflow:hidden">
            <div style="width:{bar_w}px;height:100%;background:{bar_color};transition:width 0.3s"></div>
          </div>
          <span style="color:#57606a;font-size:0.82rem">{token_used:,}&nbsp;/&nbsp;{token_budget:,} tokens</span>
        </div>
      </td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:top">Retrieved</td>
      <td style="vertical-align:top">{file_rows}</td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Strategy</td>
      <td style="color:#8250df;font-weight:bold">{strategy}</td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Prompt&nbsp;size</td>
      <td style="color:#57606a">{prompt_size:,} tokens</td>
    </tr>
  </table>
</div>"""
    display(HTML(html))


def show_reply(text: str) -> None:
    """
    Render the model's reply as markdown in the notebook output.
    Falls back to plain print() outside of Jupyter.
    """
    try:
        display(Markdown(text))
    except Exception:
        print(text)

```

### 2.5 Try it

Send a single question to Ollama and observe the panel
before the call (prompt assembled, no reply yet) and after
(real token count from Ollama). The response will be rendered as markdown just below the panel.

The token count before the call is a rough estimate (`len(text) // 4`).
From Chapter 3 onward we'll use a proper tokeniser.


```python
query    = "What is a context window, and why does its size matter?"
messages = [{"role": "user", "content": query}]

# Rough pre-call estimate: 1 token ≈ 4 characters
prompt_size = len(query) // 4

show_rule("Before LLM call")
show_panel(
    query        = query,
    token_used   = prompt_size,
    strategy     = "none",
    prompt_size  = prompt_size,
)

print("\nSending query to Ollama…\n")
reply, tokens_used = chat(messages)

show_rule("After LLM call")
show_panel(
    query       = query,
    token_used  = tokens_used,
    strategy    = "none",
    prompt_size = prompt_size,
)

show_reply(reply)

```


<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">Before LLM call</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>




<div style="border:1px solid #d0d7de;border-radius:8px;padding:14px 18px;
            margin:8px 0;background:#f6f8fa;font-family:monospace;font-size:0.88rem">
  <div style="font-weight:bold;color:#0969da;margin-bottom:10px">
    Pocket Agent&nbsp;&nbsp;·&nbsp;&nbsp;<em style="font-weight:normal;color:#57606a">What is a context window, and why does its size matter?</em>
  </div>
  <table style="border-collapse:collapse;width:100%">
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:middle;white-space:nowrap">Token Budget</td>
      <td style="vertical-align:middle">
        <div style="display:inline-flex;align-items:center;gap:10px">
          <div style="width:200px;height:10px;border-radius:5px;background:#e0e0e0;overflow:hidden">
            <div style="width:0px;height:100%;background:#2da44e;transition:width 0.3s"></div>
          </div>
          <span style="color:#57606a;font-size:0.82rem">13&nbsp;/&nbsp;262,144 tokens</span>
        </div>
      </td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:top">Retrieved</td>
      <td style="vertical-align:top"><span style="color:#8c959f;font-style:italic">none</span></td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Strategy</td>
      <td style="color:#8250df;font-weight:bold">none</td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Prompt&nbsp;size</td>
      <td style="color:#57606a">13 tokens</td>
    </tr>
  </table>
</div>


    
    Sending query to Ollama…
    



<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">After LLM call</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>




<div style="border:1px solid #d0d7de;border-radius:8px;padding:14px 18px;
            margin:8px 0;background:#f6f8fa;font-family:monospace;font-size:0.88rem">
  <div style="font-weight:bold;color:#0969da;margin-bottom:10px">
    Pocket Agent&nbsp;&nbsp;·&nbsp;&nbsp;<em style="font-weight:normal;color:#57606a">What is a context window, and why does its size matter?</em>
  </div>
  <table style="border-collapse:collapse;width:100%">
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:middle;white-space:nowrap">Token Budget</td>
      <td style="vertical-align:middle">
        <div style="display:inline-flex;align-items:center;gap:10px">
          <div style="width:200px;height:10px;border-radius:5px;background:#e0e0e0;overflow:hidden">
            <div style="width:0px;height:100%;background:#2da44e;transition:width 0.3s"></div>
          </div>
          <span style="color:#57606a;font-size:0.82rem">592&nbsp;/&nbsp;262,144 tokens</span>
        </div>
      </td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:top">Retrieved</td>
      <td style="vertical-align:top"><span style="color:#8c959f;font-style:italic">none</span></td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Strategy</td>
      <td style="color:#8250df;font-weight:bold">none</td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Prompt&nbsp;size</td>
      <td style="color:#57606a">13 tokens</td>
    </tr>
  </table>
</div>



A **context window** refers to the maximum amount of text (tokens) that a language model can process at once—both as input (prompt) and output (response)—during a single interaction.

### Key Details:
- **Tokens**: The basic unit (e.g., words, subwords, punctuation). 1 token ≈ 0.75 English words on average.
- **Window size**: Typically measured in tokens (e.g., 4K, 8K, 32K, 128K, or even 2M+ for newer models like Claude 3.5 Sonnet or Gemini 1.5 Pro).
- **How it works**:
  - When you send a prompt and request a response, the model uses tokens up to its context limit.
  - If the total (prompt + response) exceeds this limit, the model truncates or rejects the request.
  - Some systems (e.g., chat interfaces) may *automatically* discard older messages to stay within the window—this is called **sliding window context**.

---

### Why Size Matters:
| Impact Area | Small Context Window (<8K) | Large Context Window (>100K) |
|-------------|----------------------------|-------------------------------|
| **Long-form tasks** | Struggles with summarizing long docs, legal contracts, or multi-chapter novels | Can process entire books or APIs documentation in one go |
| **Multitasking** | Limited ability to juggle multiple topics or sources | Can reference many files, chat history, or codebases simultaneously |
| **Temporal awareness** | May forget early parts of long conversations | Maintains full dialogue context (e.g., remembers your first question) |
| **Accuracy & coherence** | Higher risk of inconsistency over long inputs | Reduces “amnesia,” supports complex reasoning across docs |
| **Practical use cases** | Chat, short summaries, Q&A with brief docs | Enterprise RAG, codebase analysis, detailed reporting, archival QA |

---

### Example:
- A model with a **32K-token** window can handle ~24,000 words in one go—enough for a full novel *or* 300+ pages of dense PDF text.
- A **4K-token** model (~3,000 words) would truncate that same novel into 2–3 pages, losing critical context.

> 💡 **Trade-off**: Larger context windows demand more memory/compute and can increase latency—but modern systems use clever techniques (like attention compression) to make them practical.

In short: **The context window defines how “long-term” a model’s memory is.** Bigger isn’t always better, but for complex, long-horizon tasks, it’s transformative.


---
## Chapter 3 — Context is a Budget

**Goal:** understand *why* context management exists before we build any of it.

Every token you load — user message, system prompt, file content, conversation history — occupies space in the context window. When that space runs out, one of two things happens: the model silently truncates early content (it forgets the beginning of the conversation), or the API returns an error.

A coding agent that naively loads every file in a repo will hit the wall on the first non-trivial query.

**You will:**
- Replace the inline `len // 4` guess with a named `count_tokens()` function
- Watch a multi-turn conversation burn through its budget turn by turn
- Measure the token cost of real files on disk
- See the panel turn yellow then red as budget pressure rises
- Understand the three strategies agents use to stay inside the window (foreshadowing Ch 3–8)

### 3.1 `count_tokens()` — one place to estimate token cost

If we were using OpenAI models, we could use tiktoken for exact token counts.
For a provider‑agnostic approach, we’d need a tokenizer from the model’s ecosystem (often a Hugging Face tokenizer) — which is heavier to load and configure.

The `÷ 4` heuristic (1 token ≈ 4 characters in English prose) is accurate to
within ~15 % for most code and documentation. That's precise enough to drive
a budget bar — we don't need exact counts, we need *early warning*.

Wrapping it in a named function means we can swap the implementation once
(e.g. call Ollama's `/api/tokenize` endpoint) without touching any of the
chapter code that calls it.


```python
def count_tokens(text: str) -> int:
    """
    Estimate token count for *text*.

    Uses the 4-characters-per-token heuristic — accurate to ~15 % for
    English prose and source code.  Good enough to drive a budget bar.

    Swap the body for a real tokeniser call if you need precision:
        # example: use Ollama's tokenise endpoint
        # r = requests.post(f"{OLLAMA_BASE_URL}/api/tokenize",
        #                   json={"model": OLLAMA_MODEL, "content": text})
        # return len(r.json()["tokens"])
    """
    return max(1, len(text) // 4)


# Quick sanity check
_sample = "The quick brown fox jumps over the lazy dog."
print(f"Sample: {len(_sample)} chars → {count_tokens(_sample)} tokens  "
      f"(GPT-4 tokeniser gives ~11 for this sentence)")
```

    Sample: 44 chars → 11 tokens  (GPT-4 tokeniser gives ~11 for this sentence)


### 3.2 Watching a conversation burn its budget

Each turn appends the user message *and* the assistant reply to the
`messages` list before the next call.  Ollama sees the full list every time —
that's how it maintains conversation context, but it also means every reply
you get back is added to your next prompt.

Run this cell and watch the budget bar grow with each exchange.


```python
TURNS = [
    "What is a Python list comprehension? Give a one-line example.",
    "Now show a dictionary comprehension that squares numbers 1–5.",
    "What is the difference between a shallow copy and a deep copy?",
    "Show me a minimal example where a shallow copy causes a bug.",
]

messages: list[dict] = []
tokens_used = 0

for i, question in enumerate(TURNS, start=1):
    messages.append({"role": "user", "content": question})

    # Estimate prompt size before the call
    prompt_text = " ".join(m["content"] for m in messages)
    prompt_size = count_tokens(prompt_text)

    show_rule(f"Turn {i} — before call")
    show_panel(
        query        = question,
        token_used   = tokens_used,
        strategy     = "multi-turn",
        prompt_size  = prompt_size,
    )

    reply, tokens_used = chat(messages)
    messages.append({"role": "assistant", "content": reply})

    show_rule(f"Turn {i} — after call")
    show_panel(
        query       = question,
        token_used  = tokens_used,
        strategy    = "multi-turn",
        prompt_size = prompt_size,
    )
    print(f"\nReply: {reply[:200]}{'…' if len(reply) > 200 else ''}\n")

```


<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">Turn 1 — before call</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>




<div style="border:1px solid #d0d7de;border-radius:8px;padding:14px 18px;
            margin:8px 0;background:#f6f8fa;font-family:monospace;font-size:0.88rem">
  <div style="font-weight:bold;color:#0969da;margin-bottom:10px">
    Pocket Agent&nbsp;&nbsp;·&nbsp;&nbsp;<em style="font-weight:normal;color:#57606a">What is a Python list comprehension? Give a one-line example.</em>
  </div>
  <table style="border-collapse:collapse;width:100%">
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:middle;white-space:nowrap">Token Budget</td>
      <td style="vertical-align:middle">
        <div style="display:inline-flex;align-items:center;gap:10px">
          <div style="width:200px;height:10px;border-radius:5px;background:#e0e0e0;overflow:hidden">
            <div style="width:0px;height:100%;background:#2da44e;transition:width 0.3s"></div>
          </div>
          <span style="color:#57606a;font-size:0.82rem">0&nbsp;/&nbsp;262,144 tokens</span>
        </div>
      </td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:top">Retrieved</td>
      <td style="vertical-align:top"><span style="color:#8c959f;font-style:italic">none</span></td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Strategy</td>
      <td style="color:#8250df;font-weight:bold">multi-turn</td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Prompt&nbsp;size</td>
      <td style="color:#57606a">15 tokens</td>
    </tr>
  </table>
</div>



<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">Turn 1 — after call</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>




<div style="border:1px solid #d0d7de;border-radius:8px;padding:14px 18px;
            margin:8px 0;background:#f6f8fa;font-family:monospace;font-size:0.88rem">
  <div style="font-weight:bold;color:#0969da;margin-bottom:10px">
    Pocket Agent&nbsp;&nbsp;·&nbsp;&nbsp;<em style="font-weight:normal;color:#57606a">What is a Python list comprehension? Give a one-line example.</em>
  </div>
  <table style="border-collapse:collapse;width:100%">
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:middle;white-space:nowrap">Token Budget</td>
      <td style="vertical-align:middle">
        <div style="display:inline-flex;align-items:center;gap:10px">
          <div style="width:200px;height:10px;border-radius:5px;background:#e0e0e0;overflow:hidden">
            <div style="width:0px;height:100%;background:#2da44e;transition:width 0.3s"></div>
          </div>
          <span style="color:#57606a;font-size:0.82rem">94&nbsp;/&nbsp;262,144 tokens</span>
        </div>
      </td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:top">Retrieved</td>
      <td style="vertical-align:top"><span style="color:#8c959f;font-style:italic">none</span></td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Strategy</td>
      <td style="color:#8250df;font-weight:bold">multi-turn</td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Prompt&nbsp;size</td>
      <td style="color:#57606a">15 tokens</td>
    </tr>
  </table>
</div>


    
    Reply: A Python list comprehension is a concise syntax for creating a new list by applying an expression to each item in an iterable, optionally filtering items with a condition.
    
    **Example:**  
    ```python
    sq…
    



<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">Turn 2 — before call</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>




<div style="border:1px solid #d0d7de;border-radius:8px;padding:14px 18px;
            margin:8px 0;background:#f6f8fa;font-family:monospace;font-size:0.88rem">
  <div style="font-weight:bold;color:#0969da;margin-bottom:10px">
    Pocket Agent&nbsp;&nbsp;·&nbsp;&nbsp;<em style="font-weight:normal;color:#57606a">Now show a dictionary comprehension that squares numbers 1–5.</em>
  </div>
  <table style="border-collapse:collapse;width:100%">
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:middle;white-space:nowrap">Token Budget</td>
      <td style="vertical-align:middle">
        <div style="display:inline-flex;align-items:center;gap:10px">
          <div style="width:200px;height:10px;border-radius:5px;background:#e0e0e0;overflow:hidden">
            <div style="width:0px;height:100%;background:#2da44e;transition:width 0.3s"></div>
          </div>
          <span style="color:#57606a;font-size:0.82rem">94&nbsp;/&nbsp;262,144 tokens</span>
        </div>
      </td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:top">Retrieved</td>
      <td style="vertical-align:top"><span style="color:#8c959f;font-style:italic">none</span></td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Strategy</td>
      <td style="color:#8250df;font-weight:bold">multi-turn</td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Prompt&nbsp;size</td>
      <td style="color:#57606a">97 tokens</td>
    </tr>
  </table>
</div>



<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">Turn 2 — after call</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>




<div style="border:1px solid #d0d7de;border-radius:8px;padding:14px 18px;
            margin:8px 0;background:#f6f8fa;font-family:monospace;font-size:0.88rem">
  <div style="font-weight:bold;color:#0969da;margin-bottom:10px">
    Pocket Agent&nbsp;&nbsp;·&nbsp;&nbsp;<em style="font-weight:normal;color:#57606a">Now show a dictionary comprehension that squares numbers 1–5.</em>
  </div>
  <table style="border-collapse:collapse;width:100%">
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:middle;white-space:nowrap">Token Budget</td>
      <td style="vertical-align:middle">
        <div style="display:inline-flex;align-items:center;gap:10px">
          <div style="width:200px;height:10px;border-radius:5px;background:#e0e0e0;overflow:hidden">
            <div style="width:0px;height:100%;background:#2da44e;transition:width 0.3s"></div>
          </div>
          <span style="color:#57606a;font-size:0.82rem">176&nbsp;/&nbsp;262,144 tokens</span>
        </div>
      </td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:top">Retrieved</td>
      <td style="vertical-align:top"><span style="color:#8c959f;font-style:italic">none</span></td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Strategy</td>
      <td style="color:#8250df;font-weight:bold">multi-turn</td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Prompt&nbsp;size</td>
      <td style="color:#57606a">97 tokens</td>
    </tr>
  </table>
</div>


    
    Reply: ```python
    squares_dict = {x: x**2 for x in range(1, 6)}  # creates {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
    ```
    



<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">Turn 3 — before call</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>




<div style="border:1px solid #d0d7de;border-radius:8px;padding:14px 18px;
            margin:8px 0;background:#f6f8fa;font-family:monospace;font-size:0.88rem">
  <div style="font-weight:bold;color:#0969da;margin-bottom:10px">
    Pocket Agent&nbsp;&nbsp;·&nbsp;&nbsp;<em style="font-weight:normal;color:#57606a">What is the difference between a shallow copy and a deep copy?</em>
  </div>
  <table style="border-collapse:collapse;width:100%">
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:middle;white-space:nowrap">Token Budget</td>
      <td style="vertical-align:middle">
        <div style="display:inline-flex;align-items:center;gap:10px">
          <div style="width:200px;height:10px;border-radius:5px;background:#e0e0e0;overflow:hidden">
            <div style="width:0px;height:100%;background:#2da44e;transition:width 0.3s"></div>
          </div>
          <span style="color:#57606a;font-size:0.82rem">176&nbsp;/&nbsp;262,144 tokens</span>
        </div>
      </td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:top">Retrieved</td>
      <td style="vertical-align:top"><span style="color:#8c959f;font-style:italic">none</span></td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Strategy</td>
      <td style="color:#8250df;font-weight:bold">multi-turn</td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Prompt&nbsp;size</td>
      <td style="color:#57606a">138 tokens</td>
    </tr>
  </table>
</div>



<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">Turn 3 — after call</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>




<div style="border:1px solid #d0d7de;border-radius:8px;padding:14px 18px;
            margin:8px 0;background:#f6f8fa;font-family:monospace;font-size:0.88rem">
  <div style="font-weight:bold;color:#0969da;margin-bottom:10px">
    Pocket Agent&nbsp;&nbsp;·&nbsp;&nbsp;<em style="font-weight:normal;color:#57606a">What is the difference between a shallow copy and a deep copy?</em>
  </div>
  <table style="border-collapse:collapse;width:100%">
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:middle;white-space:nowrap">Token Budget</td>
      <td style="vertical-align:middle">
        <div style="display:inline-flex;align-items:center;gap:10px">
          <div style="width:200px;height:10px;border-radius:5px;background:#e0e0e0;overflow:hidden">
            <div style="width:0px;height:100%;background:#2da44e;transition:width 0.3s"></div>
          </div>
          <span style="color:#57606a;font-size:0.82rem">387&nbsp;/&nbsp;262,144 tokens</span>
        </div>
      </td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:top">Retrieved</td>
      <td style="vertical-align:top"><span style="color:#8c959f;font-style:italic">none</span></td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Strategy</td>
      <td style="color:#8250df;font-weight:bold">multi-turn</td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Prompt&nbsp;size</td>
      <td style="color:#57606a">138 tokens</td>
    </tr>
  </table>
</div>


    
    Reply: A **shallow copy** creates a new container (e.g., list, dict), but *copies references* to the nested objects—so changes to nested mutable objects affect both the original and the copy.  
    A **deep copy…
    



<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">Turn 4 — before call</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>




<div style="border:1px solid #d0d7de;border-radius:8px;padding:14px 18px;
            margin:8px 0;background:#f6f8fa;font-family:monospace;font-size:0.88rem">
  <div style="font-weight:bold;color:#0969da;margin-bottom:10px">
    Pocket Agent&nbsp;&nbsp;·&nbsp;&nbsp;<em style="font-weight:normal;color:#57606a">Show me a minimal example where a shallow copy causes a bug.</em>
  </div>
  <table style="border-collapse:collapse;width:100%">
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:middle;white-space:nowrap">Token Budget</td>
      <td style="vertical-align:middle">
        <div style="display:inline-flex;align-items:center;gap:10px">
          <div style="width:200px;height:10px;border-radius:5px;background:#e0e0e0;overflow:hidden">
            <div style="width:0px;height:100%;background:#2da44e;transition:width 0.3s"></div>
          </div>
          <span style="color:#57606a;font-size:0.82rem">387&nbsp;/&nbsp;262,144 tokens</span>
        </div>
      </td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:top">Retrieved</td>
      <td style="vertical-align:top"><span style="color:#8c959f;font-style:italic">none</span></td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Strategy</td>
      <td style="color:#8250df;font-weight:bold">multi-turn</td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Prompt&nbsp;size</td>
      <td style="color:#57606a">326 tokens</td>
    </tr>
  </table>
</div>



<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">Turn 4 — after call</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>




<div style="border:1px solid #d0d7de;border-radius:8px;padding:14px 18px;
            margin:8px 0;background:#f6f8fa;font-family:monospace;font-size:0.88rem">
  <div style="font-weight:bold;color:#0969da;margin-bottom:10px">
    Pocket Agent&nbsp;&nbsp;·&nbsp;&nbsp;<em style="font-weight:normal;color:#57606a">Show me a minimal example where a shallow copy causes a bug.</em>
  </div>
  <table style="border-collapse:collapse;width:100%">
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:middle;white-space:nowrap">Token Budget</td>
      <td style="vertical-align:middle">
        <div style="display:inline-flex;align-items:center;gap:10px">
          <div style="width:200px;height:10px;border-radius:5px;background:#e0e0e0;overflow:hidden">
            <div style="width:0px;height:100%;background:#2da44e;transition:width 0.3s"></div>
          </div>
          <span style="color:#57606a;font-size:0.82rem">586&nbsp;/&nbsp;262,144 tokens</span>
        </div>
      </td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:top">Retrieved</td>
      <td style="vertical-align:top"><span style="color:#8c959f;font-style:italic">none</span></td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Strategy</td>
      <td style="color:#8250df;font-weight:bold">multi-turn</td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Prompt&nbsp;size</td>
      <td style="color:#57606a">326 tokens</td>
    </tr>
  </table>
</div>


    
    Reply: Here's a minimal example where a shallow copy causes an unintended side effect:
    
    ```python
    # Original list containing a mutable nested list
    data = [1, 2, [3, 4]]
    shallow_copy = data.copy()  # shallow …
    


### 3.3 What files actually cost

A coding agent loads source files to answer questions about them.
Let's measure what that costs in tokens.

We'll scan the files in `REPO_ROOT`, count their token cost, and show
how quickly a naive "load everything" strategy would exhaust the budget.


```python
import os
from pathlib import Path

# File extensions we'd want to load for a coding question
CODE_EXTENSIONS = {".py", ".js", ".ts", ".go", ".rs", ".java",
                   ".c", ".cpp", ".h", ".md", ".txt", ".yaml", ".toml", ".json"}

def scan_repo_costs(root: str) -> list[dict]:
    """
    Walk *root* and return a list of dicts:
        {"path": relative_path, "bytes": int, "tokens": int}
    for every source file found, sorted by token cost descending.
    """
    results = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if not d.startswith(".") and d not in
                       {"__pycache__", "node_modules", ".git", "venv", ".venv"}]
        for fname in filenames:
            full = Path(dirpath) / fname
            if full.suffix.lower() not in CODE_EXTENSIONS:
                continue
            try:
                text = full.read_text(errors="ignore")
            except OSError:
                continue
            results.append({
                "path":   str(full.relative_to(root)),
                "bytes":  len(text.encode()),
                "tokens": count_tokens(text),
            })
    return sorted(results, key=lambda x: x["tokens"], reverse=True)


file_costs = scan_repo_costs(REPO_ROOT)

# ── HTML table ──────────────────────────────────────────────────────────────
rows = ""
for f in file_costs[:20]:
    pct = f["tokens"] / TOKEN_BUDGET * 100
    color = "#2da44e" if pct < 10 else ("#bf8700" if pct < 30 else "#cf222e")
    rows += (
        f'<tr><td style="font-family:monospace;padding:3px 12px 3px 0">{f["path"]}</td>'
        f'<td style="text-align:right;padding:3px 8px">{f["bytes"]:,}</td>'
        f'<td style="text-align:right;padding:3px 8px;color:{color};font-weight:bold">{f["tokens"]:,}</td>'
        f'<td style="text-align:right;padding:3px 8px;color:{color}">{pct:.1f}%</td></tr>'
    )

total = sum(f["tokens"] for f in file_costs)
fits  = total <= TOKEN_BUDGET
summary_color = "#2da44e" if fits else "#cf222e"
summary = "YES ✓" if fits else "NO ✗"

display(HTML(f"""
<div style="font-family:monospace;font-size:0.88rem">
  <b>File token costs in '{REPO_ROOT}'</b>
  <table style="border-collapse:collapse;margin-top:8px">
    <tr style="border-bottom:1px solid #d0d7de;color:#57606a">
      <th style="text-align:left;padding:3px 12px 3px 0">File</th>
      <th style="text-align:right;padding:3px 8px">Bytes</th>
      <th style="text-align:right;padding:3px 8px">Tokens</th>
      <th style="text-align:right;padding:3px 8px">% of budget</th>
    </tr>
    {rows}
  </table>
  <div style="margin-top:8px;color:#57606a">
    Total across {len(file_costs)} files:&nbsp;
    <span style="color:{summary_color};font-weight:bold">{total:,} tokens</span>
    &nbsp;·&nbsp; Budget: {TOKEN_BUDGET:,}
    &nbsp;·&nbsp; Fits in one prompt:&nbsp;
    <span style="color:{summary_color};font-weight:bold">{summary}</span>
  </div>
</div>
"""))

```



<div style="font-family:monospace;font-size:0.88rem">
  <b>File token costs in '.'</b>
  <table style="border-collapse:collapse;margin-top:8px">
    <tr style="border-bottom:1px solid #d0d7de;color:#57606a">
      <th style="text-align:left;padding:3px 12px 3px 0">File</th>
      <th style="text-align:right;padding:3px 8px">Bytes</th>
      <th style="text-align:right;padding:3px 8px">Tokens</th>
      <th style="text-align:right;padding:3px 8px">% of budget</th>
    </tr>
    <tr><td style="font-family:monospace;padding:3px 12px 3px 0">agent/core.py</td><td style="text-align:right;padding:3px 8px">46,140</td><td style="text-align:right;padding:3px 8px;color:#2da44e;font-weight:bold">10,431</td><td style="text-align:right;padding:3px 8px;color:#2da44e">4.0%</td></tr><tr><td style="font-family:monospace;padding:3px 12px 3px 0">agent/app.py</td><td style="text-align:right;padding:3px 8px">21,145</td><td style="text-align:right;padding:3px 8px;color:#2da44e;font-weight:bold">4,807</td><td style="text-align:right;padding:3px 8px;color:#2da44e">1.8%</td></tr><tr><td style="font-family:monospace;padding:3px 12px 3px 0">README.md</td><td style="text-align:right;padding:3px 8px">6,941</td><td style="text-align:right;padding:3px 8px;color:#2da44e;font-weight:bold">1,723</td><td style="text-align:right;padding:3px 8px;color:#2da44e">0.7%</td></tr><tr><td style="font-family:monospace;padding:3px 12px 3px 0">docs/pathlib_cheatsheet.md</td><td style="text-align:right;padding:3px 8px">1,838</td><td style="text-align:right;padding:3px 8px;color:#2da44e;font-weight:bold">459</td><td style="text-align:right;padding:3px 8px;color:#2da44e">0.2%</td></tr><tr><td style="font-family:monospace;padding:3px 12px 3px 0">AGENTS.md</td><td style="text-align:right;padding:3px 8px">1,323</td><td style="text-align:right;padding:3px 8px;color:#2da44e;font-weight:bold">326</td><td style="text-align:right;padding:3px 8px;color:#2da44e">0.1%</td></tr><tr><td style="font-family:monospace;padding:3px 12px 3px 0">sample_project/utils/parser.py</td><td style="text-align:right;padding:3px 8px">1,221</td><td style="text-align:right;padding:3px 8px;color:#2da44e;font-weight:bold">303</td><td style="text-align:right;padding:3px 8px;color:#2da44e">0.1%</td></tr><tr><td style="font-family:monospace;padding:3px 12px 3px 0">sample_project/tests/test_parser.py</td><td style="text-align:right;padding:3px 8px">1,116</td><td style="text-align:right;padding:3px 8px;color:#2da44e;font-weight:bold">279</td><td style="text-align:right;padding:3px 8px;color:#2da44e">0.1%</td></tr><tr><td style="font-family:monospace;padding:3px 12px 3px 0">sample_project/utils/validator.py</td><td style="text-align:right;padding:3px 8px">1,044</td><td style="text-align:right;padding:3px 8px;color:#2da44e;font-weight:bold">261</td><td style="text-align:right;padding:3px 8px;color:#2da44e">0.1%</td></tr><tr><td style="font-family:monospace;padding:3px 12px 3px 0">agent/README.md</td><td style="text-align:right;padding:3px 8px">1,039</td><td style="text-align:right;padding:3px 8px;color:#2da44e;font-weight:bold">259</td><td style="text-align:right;padding:3px 8px;color:#2da44e">0.1%</td></tr><tr><td style="font-family:monospace;padding:3px 12px 3px 0">sample_project/tests/test_formatter_gen.py</td><td style="text-align:right;padding:3px 8px">1,023</td><td style="text-align:right;padding:3px 8px;color:#2da44e;font-weight:bold">255</td><td style="text-align:right;padding:3px 8px;color:#2da44e">0.1%</td></tr><tr><td style="font-family:monospace;padding:3px 12px 3px 0">sample_project/utils/formatter.py</td><td style="text-align:right;padding:3px 8px">895</td><td style="text-align:right;padding:3px 8px;color:#2da44e;font-weight:bold">223</td><td style="text-align:right;padding:3px 8px;color:#2da44e">0.1%</td></tr><tr><td style="font-family:monospace;padding:3px 12px 3px 0">sample_project/main.py</td><td style="text-align:right;padding:3px 8px">694</td><td style="text-align:right;padding:3px 8px;color:#2da44e;font-weight:bold">173</td><td style="text-align:right;padding:3px 8px;color:#2da44e">0.1%</td></tr><tr><td style="font-family:monospace;padding:3px 12px 3px 0">sample_project/tests/test_formatter.py</td><td style="text-align:right;padding:3px 8px">674</td><td style="text-align:right;padding:3px 8px;color:#2da44e;font-weight:bold">168</td><td style="text-align:right;padding:3px 8px;color:#2da44e">0.1%</td></tr><tr><td style="font-family:monospace;padding:3px 12px 3px 0">agent/__init__.py</td><td style="text-align:right;padding:3px 8px">0</td><td style="text-align:right;padding:3px 8px;color:#2da44e;font-weight:bold">1</td><td style="text-align:right;padding:3px 8px;color:#2da44e">0.0%</td></tr>
  </table>
  <div style="margin-top:8px;color:#57606a">
    Total across 14 files:&nbsp;
    <span style="color:#2da44e;font-weight:bold">19,668 tokens</span>
    &nbsp;·&nbsp; Budget: 262,144
    &nbsp;·&nbsp; Fits in one prompt:&nbsp;
    <span style="color:#2da44e;font-weight:bold">YES ✓</span>
  </div>
</div>



### 3.4 Hitting the wall — what over-budget looks like

Let's simulate what happens when a naive agent stuffs too much into one prompt.
We'll build a fake "load everything" prompt, measure it, and show the panel
turning red — without actually sending it to Ollama (no point wasting tokens
on a prompt that would be truncated or error).


```python
def _simulate_overbudget() -> None:
    """
    Build a hypothetical prompt that exceeds TOKEN_BUDGET and show
    the panel at 50 %, 85 %, and 110 % capacity — no LLM call needed.
    """
    # Each file is a *fraction* of the budget so accumulation crosses 100 %
    # gradually: file 1 → ~50 %, file 2 → ~85 %, file 3 → ~112 %
    fake_files = [
        {"name": "src/parser.py",       "tokens": int(TOKEN_BUDGET * 0.50)},
        {"name": "src/compiler.py",     "tokens": int(TOKEN_BUDGET * 0.35)},
        {"name": "src/runtime.py",      "tokens": int(TOKEN_BUDGET * 0.27)},
        {"name": "tests/test_parser.py","tokens": int(TOKEN_BUDGET * 0.20)},
    ]

    loaded_files = []
    accumulated  = 0
    query        = "Explain how the compiler hands off to the runtime"

    for i, f in enumerate(fake_files):
        accumulated += f["tokens"]
        loaded_files.append({"path": f["name"], "hot": True})

        pct = accumulated / TOKEN_BUDGET
        label = (
            "comfortable" if pct < 0.70 else
            "warning"     if pct < 1.00 else
            "OVER BUDGET"
        )
        show_rule(f"After loading {i+1} file(s) — {label}")
        show_panel(
            query        = query,
            token_used   = accumulated,
            files        = loaded_files,
            strategy     = "naive load-all",
            prompt_size  = accumulated,
        )

        if accumulated > TOKEN_BUDGET:
            print("\n⚠  Prompt exceeds context window.")
            print("Ollama will truncate the earliest messages — the agent may")
            print("forget the system prompt or earlier file contents entirely.\n")
            break

_simulate_overbudget()
```


<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">After loading 1 file(s) — comfortable</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>




<div style="border:1px solid #d0d7de;border-radius:8px;padding:14px 18px;
            margin:8px 0;background:#f6f8fa;font-family:monospace;font-size:0.88rem">
  <div style="font-weight:bold;color:#0969da;margin-bottom:10px">
    Pocket Agent&nbsp;&nbsp;·&nbsp;&nbsp;<em style="font-weight:normal;color:#57606a">Explain how the compiler hands off to the runtime</em>
  </div>
  <table style="border-collapse:collapse;width:100%">
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:middle;white-space:nowrap">Token Budget</td>
      <td style="vertical-align:middle">
        <div style="display:inline-flex;align-items:center;gap:10px">
          <div style="width:200px;height:10px;border-radius:5px;background:#e0e0e0;overflow:hidden">
            <div style="width:100px;height:100%;background:#2da44e;transition:width 0.3s"></div>
          </div>
          <span style="color:#57606a;font-size:0.82rem">131,072&nbsp;/&nbsp;262,144 tokens</span>
        </div>
      </td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:top">Retrieved</td>
      <td style="vertical-align:top"><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">src/parser.py</code></div></td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Strategy</td>
      <td style="color:#8250df;font-weight:bold">naive load-all</td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Prompt&nbsp;size</td>
      <td style="color:#57606a">131,072 tokens</td>
    </tr>
  </table>
</div>



<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">After loading 2 file(s) — warning</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>




<div style="border:1px solid #d0d7de;border-radius:8px;padding:14px 18px;
            margin:8px 0;background:#f6f8fa;font-family:monospace;font-size:0.88rem">
  <div style="font-weight:bold;color:#0969da;margin-bottom:10px">
    Pocket Agent&nbsp;&nbsp;·&nbsp;&nbsp;<em style="font-weight:normal;color:#57606a">Explain how the compiler hands off to the runtime</em>
  </div>
  <table style="border-collapse:collapse;width:100%">
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:middle;white-space:nowrap">Token Budget</td>
      <td style="vertical-align:middle">
        <div style="display:inline-flex;align-items:center;gap:10px">
          <div style="width:200px;height:10px;border-radius:5px;background:#e0e0e0;overflow:hidden">
            <div style="width:169px;height:100%;background:#bf8700;transition:width 0.3s"></div>
          </div>
          <span style="color:#57606a;font-size:0.82rem">222,822&nbsp;/&nbsp;262,144 tokens</span>
        </div>
      </td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:top">Retrieved</td>
      <td style="vertical-align:top"><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">src/parser.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">src/compiler.py</code></div></td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Strategy</td>
      <td style="color:#8250df;font-weight:bold">naive load-all</td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Prompt&nbsp;size</td>
      <td style="color:#57606a">222,822 tokens</td>
    </tr>
  </table>
</div>



<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">After loading 3 file(s) — OVER BUDGET</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>




<div style="border:1px solid #d0d7de;border-radius:8px;padding:14px 18px;
            margin:8px 0;background:#f6f8fa;font-family:monospace;font-size:0.88rem">
  <div style="font-weight:bold;color:#0969da;margin-bottom:10px">
    Pocket Agent&nbsp;&nbsp;·&nbsp;&nbsp;<em style="font-weight:normal;color:#57606a">Explain how the compiler hands off to the runtime</em>
  </div>
  <table style="border-collapse:collapse;width:100%">
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:middle;white-space:nowrap">Token Budget</td>
      <td style="vertical-align:middle">
        <div style="display:inline-flex;align-items:center;gap:10px">
          <div style="width:200px;height:10px;border-radius:5px;background:#e0e0e0;overflow:hidden">
            <div style="width:200px;height:100%;background:#cf222e;transition:width 0.3s"></div>
          </div>
          <span style="color:#57606a;font-size:0.82rem">293,600&nbsp;/&nbsp;262,144 tokens</span>
        </div>
      </td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:top">Retrieved</td>
      <td style="vertical-align:top"><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">src/parser.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">src/compiler.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">src/runtime.py</code></div></td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Strategy</td>
      <td style="color:#8250df;font-weight:bold">naive load-all</td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Prompt&nbsp;size</td>
      <td style="color:#57606a">293,600 tokens</td>
    </tr>
  </table>
</div>


    
    ⚠  Prompt exceeds context window.
    Ollama will truncate the earliest messages — the agent may
    forget the system prompt or earlier file contents entirely.
    


### 3.5 The three strategies — what's coming next

Every technique in Chapters 3–8 is a variation of one of these three responses
to budget pressure:

| Strategy | What it does | Where we build it |
|----------|-------------|-------------------|
| **Be selective** | Only load files relevant to the query | Ch 3 (manifest), Ch 4 (glob), Ch 5 (fuzzy), Ch 6 (grep) |
| **Be lazy** | Load file *summaries* first, full content only on demand | Ch 4 JIT reads |
| **Evict & compress** | Move cold context to a summary store; keep hot context small | Ch 7 microcompaction, Ch 8 semantic RAG |

The key insight: **a good agent spends tokens like a good programmer spends CPU** — only when necessary, and on the thing most likely to matter.

Chapter 4 introduces the first tool: an `AGENTS.md` manifest that tells the agent which files matter *before* it even looks at the file tree.

---
## Chapter 4 — Give it a Map

**Goal:** before the agent looks at a single file, hand it a curated index of the repo so it already knows what exists and what matters.

That index is `AGENTS.md` — a small markdown file you commit at the repo root.
It costs a fixed, known number of tokens on every call (cheap), and it dramatically reduces the search space for every retrieval strategy we build later.

**You will:**
- Create an `AGENTS.md` for this repo
- Write `load_manifest()` — reads the file, extracts mentioned paths
- Write `ask_with_manifest()` — prepends the manifest to every prompt
- See the panel report the manifest's token cost before any file is loaded

### 4.1 What goes in AGENTS.md?

An `AGENTS.md` is just a markdown file. It has no required schema — the agent
reads it as plain text. The convention is:

- **Entry points** — where execution starts
- **Key modules** — what each important file does in one line
- **Ownership sections** — "questions about X → look in Y"
- **Off-limits** — generated files, build artefacts the agent should skip

Keep it under ~400 tokens. If it grows larger than that, it starts eating the
budget it was meant to protect.

The cell below uses `%%writefile` to create `AGENTS.md` on disk.
`%%writefile path` is a Jupyter magic: instead of running the cell,
it saves the cell body verbatim to the given path. The file will live next to
this notebook in `REPO_ROOT`.


```python
%%writefile AGENTS.md
# AGENTS.md — Pocket Agent project map

## What this repo is
A Jupyter notebook tutorial that builds a local coding agent step by step.
Each chapter adds one capability. The notebook IS the source of truth.

## Entry point
- `pocket_agent.ipynb` — the entire project lives here

## Chapter guide
| Chapter | Topic | Key concepts defined |
|---------|-------|----------------------|
| 1 | Hello Ollama | `chat()`, `show_panel()`, `ping_ollama()` |
| 2 | Context budget | `count_tokens()`, `scan_repo_costs()` |
| 3 | Manifest | `load_manifest()`, `ask_with_manifest()` |
| 4 | Glob + JIT reads | `glob_files()`, `jit_read()` |
| 5 | Fuzzy scoring | `score_files()` |
| 6 | Grep | `grep_repo()` |
| 7 | Microcompaction | `compact()`, hot/cold store |
| 8 | Semantic RAG | `embed()`, `retrieve()` |
| 9 | Full pipeline | `run()` |
| 9b | Web UI | FastAPI + WebSocket server |
| 10 | Write + diff | `write_file()`, `make_diff()` |
| 11 | Agent loop | `agent_loop()` |
| 12 | Test generation | `generate_tests()` |

## Off-limits (never load these)
- `.git/` — git internals
- `__pycache__/` — compiled bytecode
- `*.ipynb_checkpoints/` — Jupyter autosave noise

## Questions about token budgeting → Chapter 3
## Questions about retrieval strategy → Chapters 4–8
## Questions about the agent loop → Chapter 12
```

    Overwriting AGENTS.md


### 4.2 `load_manifest()` — read the map

`load_manifest()` does two things:
1. Reads `AGENTS.md` as plain text (to inject into the prompt)
2. Extracts any file paths mentioned in it (for later retrieval stages to use as hints)

The path extraction is intentionally simple — a regex that finds things
that look like `path/to/file.ext`. False positives are fine; this is
advisory, not authoritative.


```python
import re

MANIFEST_FILENAME = "AGENTS.md"

def load_manifest(repo_root: str = REPO_ROOT) -> dict:
    """
    Read AGENTS.md from *repo_root*.

    Returns a dict:
        {
          "text":   str,        # full file content, ready to inject into a prompt
          "tokens": int,        # estimated token cost
          "paths":  list[str],  # file paths mentioned in the manifest
          "found":  bool,       # False if the file doesn't exist
        }
    """
    manifest_path = Path(repo_root) / MANIFEST_FILENAME

    if not manifest_path.exists():
        return {"text": "", "tokens": 0, "paths": [], "found": False}

    text = manifest_path.read_text(errors="ignore")

    # Extract things that look like file paths: word chars, slashes, dots
    # e.g.  pocket_agent.ipynb  src/parser.py  docs/architecture.md
    paths = re.findall(r'\b[\w./\-]+\.[\w]+\b', text)
    # Deduplicate while preserving order
    seen, unique_paths = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)

    return {
        "text":   text,
        "tokens": count_tokens(text),
        "paths":  unique_paths,
        "found":  True,
    }


manifest = load_manifest()
print(f"Manifest found : {manifest['found']}")
print(f"Token cost     : {manifest['tokens']}  ({manifest['tokens']/TOKEN_BUDGET*100:.1f}% of budget)")
print(f"Paths mentioned: {manifest['paths'][:8]}")
```

    Manifest found : True
    Token cost     : 326  (0.1% of budget)
    Paths mentioned: ['AGENTS.md', 'pocket_agent.ipynb']


### 4.3 `ask_with_manifest()` — the manifest-aware query

Every prompt the agent sends from now on follows this structure:

```
[SYSTEM]
You are a coding assistant. Here is the project map:
<AGENTS.md contents>

[USER]
<query>
```

The manifest is injected once, costs a fixed number of tokens, and gives
the model the project layout before it has to reason about anything.


```python
def ask_with_manifest(
    query:     str,
    repo_root: str = REPO_ROOT,
    files:     list[dict] | None = None,
) -> tuple[str, int]:
    """
    Send *query* to Ollama with the project manifest prepended as a system prompt.

    Parameters
    ----------
    query     : the user's question
    repo_root : where to find AGENTS.md
    files     : already-loaded file dicts {"path", "content", "hot"}
                their content is appended after the manifest

    Returns (reply_text, tokens_used).
    """
    manifest = load_manifest(repo_root)
    files    = files or []

    # Build system prompt
    if manifest["found"]:
        system_content = (
            "You are a coding assistant with access to the project map below.\n"
            "Use it to understand the codebase structure before answering.\n\n"
            f"--- PROJECT MAP (AGENTS.md) ---\n{manifest['text']}\n---"
        )
    else:
        system_content = "You are a coding assistant."

    # Append any loaded file contents
    file_block = ""
    for f in files:
        file_block += f"\n\n--- FILE: {f['path']} ---\n{f.get('content', '')}"

    messages = [
        {"role": "system",  "content": system_content},
        {"role": "user",    "content": query + file_block},
    ]

    prompt_size = count_tokens(system_content + query + file_block)

    show_panel(
        query        = query,
        token_used   = prompt_size,
        files        = [{"path": f["path"], "hot": f.get("hot", True)} for f in files],
        strategy     = "manifest",
        prompt_size  = prompt_size,
    )

    reply, tokens_used = chat(messages)
    return reply, tokens_used
```

### 4.4 Try it

Ask a question about the project. The manifest is in the prompt so the model
already knows the chapter structure before it replies.


```python
query = "Which chapter should I look at to understand how retrieval works?"

show_rule("Chapter 4 — manifest-guided query")
reply, tokens_used = ask_with_manifest(query)

show_rule("After call")
show_panel(
    query       = query,
    token_used  = tokens_used,
    strategy    = "manifest",
    prompt_size = count_tokens(query),
)
show_reply(reply)

```


<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">Chapter 4 — manifest-guided query</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>




<div style="border:1px solid #d0d7de;border-radius:8px;padding:14px 18px;
            margin:8px 0;background:#f6f8fa;font-family:monospace;font-size:0.88rem">
  <div style="font-weight:bold;color:#0969da;margin-bottom:10px">
    Pocket Agent&nbsp;&nbsp;·&nbsp;&nbsp;<em style="font-weight:normal;color:#57606a">Which chapter should I look at to understand how retrieval works?</em>
  </div>
  <table style="border-collapse:collapse;width:100%">
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:middle;white-space:nowrap">Token Budget</td>
      <td style="vertical-align:middle">
        <div style="display:inline-flex;align-items:center;gap:10px">
          <div style="width:200px;height:10px;border-radius:5px;background:#e0e0e0;overflow:hidden">
            <div style="width:0px;height:100%;background:#2da44e;transition:width 0.3s"></div>
          </div>
          <span style="color:#57606a;font-size:0.82rem">383&nbsp;/&nbsp;262,144 tokens</span>
        </div>
      </td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:top">Retrieved</td>
      <td style="vertical-align:top"><span style="color:#8c959f;font-style:italic">none</span></td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Strategy</td>
      <td style="color:#8250df;font-weight:bold">manifest</td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Prompt&nbsp;size</td>
      <td style="color:#57606a">383 tokens</td>
    </tr>
  </table>
</div>



<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">After call</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>




<div style="border:1px solid #d0d7de;border-radius:8px;padding:14px 18px;
            margin:8px 0;background:#f6f8fa;font-family:monospace;font-size:0.88rem">
  <div style="font-weight:bold;color:#0969da;margin-bottom:10px">
    Pocket Agent&nbsp;&nbsp;·&nbsp;&nbsp;<em style="font-weight:normal;color:#57606a">Which chapter should I look at to understand how retrieval works?</em>
  </div>
  <table style="border-collapse:collapse;width:100%">
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:middle;white-space:nowrap">Token Budget</td>
      <td style="vertical-align:middle">
        <div style="display:inline-flex;align-items:center;gap:10px">
          <div style="width:200px;height:10px;border-radius:5px;background:#e0e0e0;overflow:hidden">
            <div style="width:0px;height:100%;background:#2da44e;transition:width 0.3s"></div>
          </div>
          <span style="color:#57606a;font-size:0.82rem">601&nbsp;/&nbsp;262,144 tokens</span>
        </div>
      </td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:top">Retrieved</td>
      <td style="vertical-align:top"><span style="color:#8c959f;font-style:italic">none</span></td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Strategy</td>
      <td style="color:#8250df;font-weight:bold">manifest</td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Prompt&nbsp;size</td>
      <td style="color:#57606a">16 tokens</td>
    </tr>
  </table>
</div>



To understand how retrieval works, look at **Chapter 8: Semantic RAG**, where the core retrieval concepts are introduced — specifically the `embed()` and `retrieve()` functions.

For additional context and how retrieval fits into the full system, you may also want to review:

- **Chapter 4 (Glob + JIT reads)** — foundational file scanning
- **Chapter 5 (Fuzzy scoring)** — file-level relevance scoring
- **Chapter 6 (Grep)** — text-based retrieval
- **Chapter 7 (Microcompaction)** — how content is prepared for retrieval
- **Chapter 9 (Full pipeline)** — how retrieval integrates into the final agent workflow

But for the *semantic retrieval* mechanism itself (i.e., vector search + embeddings), **Chapter 8 is your primary reference**.


---
## Chapter 5 — Glob + JIT Reads

**Goal:** navigate a file tree without bulk-loading it.

Chapter 4 gave the agent a curated map. But what about files the map doesn't
mention, or repos where no `AGENTS.md` exists? The agent needs to *find* files
on its own — without reading them all.

The answer is a two-phase approach:

1. **Glob** — list every file that *could* be relevant (filenames only, no content).  
   This is essentially free: no tokens spent yet.
2. **JIT read** — load each file's content *just in time*, one at a time,  
   and stop when the token budget would tip past a safe threshold.

Files that were found but not loaded appear as **COLD** in the panel.
Files whose content is in the prompt are **HOT**.

**You will:**
- Create a small sample project to give glob something to work with
- Write `glob_files()` — match file paths against a pattern
- Write `jit_read()` — load one file on demand
- Write `budget_load()` — greedily load HOT files until budget threshold
- See the panel split between HOT and COLD for the first time

### 5.1 Sample project

Our repo only has one notebook and `AGENTS.md` — not enough to demonstrate
file navigation. The cells below use `%%writefile` to create a small fake
Python project under `sample_project/`.

Run each cell once; the files persist on disk for all later chapters.


```python
from typing import NamedTuple


class RunResult(NamedTuple):
    reply:       str
    strategy:    str
    files:       list[dict]   # full list, HOT and COLD
    tokens_used: int
    compact_log: list[str]    # empty if compaction didn't fire


def run(
    query:     str,
    repo_root: str  = REPO_ROOT,
    strategy:  str  = "auto",   # "auto" | "fuzzy" | "grep" | "semantic"
    top_k:     int  = 8,
) -> RunResult:
    """
    Full retrieval + generation pipeline.

    Parameters
    ----------
    query     : the user's question
    repo_root : path to the repository to search
    strategy  : retrieval strategy; "auto" lets pick_strategy() decide
    top_k     : max candidates to consider before budget_load()
    """
    # ── 1. Manifest ─────────────────────────────────────────────────────────
    manifest     = load_manifest(repo_root)    # from the target repo, not REPO_ROOT
    manifest_tok = manifest["tokens"]

    # ── 2. Strategy selection ────────────────────────────────────────────────
    strat = pick_strategy(query) if strategy == "auto" else strategy

    # ── 3. Retrieval ─────────────────────────────────────────────────────────
    if strat == "grep":
        candidates = grep_rank(query, repo_root=repo_root)
        # Append fuzzy extras for files with zero grep hits
        found_paths = {f["path"] for f in candidates}
        extras = [f for f in rank_files(glob_files("*.py", repo_root=repo_root), query)
                  if f["path"] not in found_paths]
        candidates = (candidates + extras)[:top_k]

    elif strat == "semantic":
        candidates = semantic_retrieve(query, repo_root=repo_root, top_k=top_k)

    else:   # fuzzy
        candidates = rank_files(glob_files("*.py", repo_root=repo_root), query)[:top_k]

    # ── 4. Budget-aware JIT load ─────────────────────────────────────────────
    loaded = budget_load(candidates, already_used=manifest_tok, repo_root=repo_root)

    # ── 5. Compaction (if needed) ─────────────────────────────────────────────
    hot_tok = sum(f.get("tokens", 0) for f in loaded if f["hot"])
    total   = manifest_tok + hot_tok + count_tokens(query)
    loaded, total, compact_log = compact(loaded, query, total)

    # ── 6. Show panel ─────────────────────────────────────────────────────────
    hot_files = [f for f in loaded if f["hot"]]
    show_panel(
        query        = query,
        token_used   = total,
        files        = [{"path": f["path"], "hot": f["hot"]} for f in loaded],
        strategy     = strat,
        prompt_size  = total,
    )

    # ── 7. Generate reply ─────────────────────────────────────────────────────
    reply, tokens_used = ask_with_manifest(query, repo_root=repo_root, files=hot_files)

    return RunResult(
        reply        = reply,
        strategy     = strat,
        files        = loaded,
        tokens_used  = tokens_used,
        compact_log  = compact_log,
    )
```


```python
from pathlib import Path
Path("sample_project/utils").mkdir(parents=True, exist_ok=True)
Path("sample_project/tests").mkdir(parents=True, exist_ok=True)
```


```python
%%writefile sample_project/main.py
"""Entry point for the JSON-to-CSV converter tool."""

from utils.parser import parse_json
from utils.formatter import to_csv
from utils.validator import validate_schema
import sys


def main(input_path: str, output_path: str, schema_path: str | None = None) -> None:
    with open(input_path) as f:
        raw = f.read()

    records = parse_json(raw)

    if schema_path:
        with open(schema_path) as f:
            schema = f.read()
        validate_schema(records, schema)

    csv_text = to_csv(records)

    with open(output_path, "w") as f:
        f.write(csv_text)

    print(f"Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main(*sys.argv[1:])
```

    Overwriting sample_project/main.py



```python
%%writefile sample_project/utils/parser.py
"""Parse a JSON string into a list of flat dicts."""

import json


def parse_json(raw: str) -> list[dict]:
    """
    Accept a JSON string that is either:
    - a list of objects   → returned as-is
    - a single object     → wrapped in a list
    - a list of scalars   → each scalar wrapped as {"value": scalar}

    Raises ValueError on malformed input.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc

    if isinstance(data, list):
        return [_flatten(item) if isinstance(item, dict) else {"value": item}
                for item in data]
    if isinstance(data, dict):
        return [_flatten(data)]

    raise ValueError(f"Expected a JSON object or array, got {type(data).__name__}")


def _flatten(obj: dict, prefix: str = "", sep: str = ".") -> dict:
    """Recursively flatten nested dicts: {"a": {"b": 1}} → {"a.b": 1}."""
    result = {}
    for key, value in obj.items():
        full_key = f"{prefix}{sep}{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_flatten(value, full_key, sep))
        else:
            result[full_key] = value
    return result
```

    Overwriting sample_project/utils/parser.py



```python
%%writefile sample_project/utils/formatter.py
"""Format a list of flat dicts as CSV text."""

import csv
import io


def to_csv(records: list[dict], delimiter: str = ",") -> str:
    """
    Convert *records* (list of flat dicts) to a CSV string.

    All keys across all records are used as headers.
    Missing values are rendered as empty strings.
    """
    if not records:
        return ""

    # Collect all headers preserving first-seen order
    headers: list[str] = []
    seen: set[str] = set()
    for rec in records:
        for key in rec:
            if key not in seen:
                headers.append(key)
                seen.add(key)

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=headers,
                            delimiter=delimiter, extrasaction="ignore")
    writer.writeheader()
    for rec in records:
        writer.writerow({h: rec.get(h, "") for h in headers})

    return buf.getvalue()
```

    Overwriting sample_project/utils/formatter.py



```python
%%writefile sample_project/utils/validator.py
"""Validate a list of records against a simple JSON schema."""

import json


def validate_schema(records: list[dict], schema_raw: str) -> None:
    """
    Validate each record against *schema_raw* (a JSON object mapping
    field names to expected types: "string", "number", "boolean").

    Raises TypeError on the first violation found.
    """
    schema: dict[str, str] = json.loads(schema_raw)
    type_map = {"string": str, "number": (int, float), "boolean": bool}

    for i, record in enumerate(records):
        for field, expected_type_name in schema.items():
            if field not in record:
                raise TypeError(
                    f"Record {i}: missing required field '{field}'"
                )
            expected = type_map.get(expected_type_name)
            if expected and not isinstance(record[field], expected):
                raise TypeError(
                    f"Record {i}: field '{field}' expected {expected_type_name}, "
                    f"got {type(record[field]).__name__}"
                )
```

    Overwriting sample_project/utils/validator.py



```python
%%writefile sample_project/tests/test_parser.py
"""Unit tests for utils/parser.py."""

import pytest
from utils.parser import parse_json, _flatten


class TestParseJson:
    def test_list_of_dicts(self):
        result = parse_json('[{"a": 1}, {"a": 2}]')
        assert result == [{"a": 1}, {"a": 2}]

    def test_single_dict_wrapped(self):
        result = parse_json('{"x": 10}')
        assert result == [{"x": 10}]

    def test_list_of_scalars(self):
        result = parse_json('[1, 2, 3]')
        assert result == [{"value": 1}, {"value": 2}, {"value": 3}]

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_json("{not valid}")

    def test_unexpected_type_raises(self):
        with pytest.raises(ValueError, match="Expected"):
            parse_json('"just a string"')


class TestFlatten:
    def test_nested_dict(self):
        assert _flatten({"a": {"b": 1}}) == {"a.b": 1}

    def test_deeply_nested(self):
        assert _flatten({"a": {"b": {"c": 42}}}) == {"a.b.c": 42}

    def test_flat_dict_unchanged(self):
        assert _flatten({"x": 1, "y": 2}) == {"x": 1, "y": 2}
```

    Overwriting sample_project/tests/test_parser.py



```python
%%writefile sample_project/tests/test_formatter.py
"""Unit tests for utils/formatter.py."""

from utils.formatter import to_csv


def test_empty_input():
    assert to_csv([]) == ""


def test_single_record():
    result = to_csv([{"name": "Alice", "age": 30}])
    lines = result.strip().splitlines()
    assert lines[0] == "name,age"
    assert lines[1] == "Alice,30"


def test_missing_fields_become_empty():
    records = [{"a": 1, "b": 2}, {"a": 3}]
    result = to_csv(records)
    lines = result.strip().splitlines()
    assert lines[0] == "a,b"
    assert lines[2] == "3,"


def test_header_order_follows_first_seen():
    records = [{"z": 1, "a": 2}]
    result = to_csv(records)
    assert result.startswith("z,a")
```

    Overwriting sample_project/tests/test_formatter.py


### 5.2 `glob_files()` — list without loading

`glob_files()` returns a list of matching file paths and their sizes —
but **no file content**. Zero tokens spent so far.

The `pattern` argument follows Python's `fnmatch` syntax:  
`*.py` matches any `.py` file, `utils/*.py` matches only files directly in `utils/`.


```python
import fnmatch

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv",
             ".ipynb_checkpoints", ".mypy_cache", ".pytest_cache"}


def glob_files(
    pattern:   str,
    repo_root: str = REPO_ROOT,
) -> list[dict]:
    """
    Walk *repo_root* and return every file whose name matches *pattern*.

    Returns a list of dicts:
        {"path": str,   # relative to repo_root
         "bytes": int,  # file size — no content loaded
         "hot":   bool} # always False at this stage
    Sorted by path for deterministic ordering.
    """
    matches = []
    root = Path(repo_root)

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for fname in filenames:
            if fnmatch.fnmatch(fname, pattern):
                full = Path(dirpath) / fname
                try:
                    size = full.stat().st_size
                except OSError:
                    continue
                matches.append({
                    "path":  str(full.relative_to(root)),
                    "bytes": size,
                    "hot":   False,
                })

    return sorted(matches, key=lambda x: x["path"])


# ── Demo ────────────────────────────────────────────────────────────────────
found = glob_files("*.py", repo_root="sample_project")
print(f"Found {len(found)} Python files:\n")
for f in found:
    print(f"  {f['path']:45s}  {f['bytes']:>5} bytes")
```

    Found 7 Python files:
    
      main.py                                          694 bytes
      tests/test_formatter.py                          674 bytes
      tests/test_formatter_gen.py                     1023 bytes
      tests/test_parser.py                            1116 bytes
      utils/formatter.py                               895 bytes
      utils/parser.py                                 1221 bytes
      utils/validator.py                              1044 bytes


### 5.3 `jit_read()` — load one file on demand

`jit_read()` takes a path from the glob list and reads its content.
It returns the same dict shape, extended with `"content"` and `"tokens"`,
and flips `"hot"` to `True`.


```python
def jit_read(file_meta: dict, repo_root: str = REPO_ROOT) -> dict:
    """
    Read the content of one file described by *file_meta* (a glob result dict).

    Returns a new dict with the same keys plus:
        "content": str   — full file text
        "tokens":  int   — estimated token cost
        "hot":     True  — this file is now in the prompt
    """
    full_path = Path(repo_root) / file_meta["path"]
    try:
        content = full_path.read_text(errors="ignore")
    except OSError as exc:
        content = f"[error reading file: {exc}]"

    return {
        **file_meta,
        "content": content,
        "tokens":  count_tokens(content),
        "hot":     True,
    }


# ── Demo: read one file ─────────────────────────────────────────────────────
sample_meta = glob_files("parser.py", repo_root="sample_project")[0]
loaded      = jit_read(sample_meta, repo_root="sample_project")

print(f"Path   : {loaded['path']}")
print(f"Tokens : {loaded['tokens']}")
print(f"Hot    : {loaded['hot']}")
print(f"\nFirst 3 lines:\n{''.join(loaded['content'].splitlines(keepends=True)[:3])}")
```

    Path   : utils/parser.py
    Tokens : 303
    Hot    : True
    
    First 3 lines:
    """Parse a JSON string into a list of flat dicts."""
    
    import json
    


### 5.4 `budget_load()` — greedy HOT/COLD split

Now we combine glob and JIT read.
`budget_load()` takes a list of glob results, reads them one by one,
and stops before the token budget hits a safety threshold.
Files it couldn't fit are kept in the list as COLD.

The threshold is 1.7 (70 % of budget) — leaving room for the manifest,
the query, and the model's reply.


```python
HOT_THRESHOLD = 0.7   # stop loading when prompt would exceed 70 % of budget


def budget_load(
    candidates: list[dict],
    already_used: int      = 0,
    repo_root:    str      = REPO_ROOT,
    threshold:    float    = HOT_THRESHOLD,
) -> list[dict]:
    """
    JIT-read files from *candidates* until adding the next one would push
    the running token total past *threshold* × TOKEN_BUDGET.

    Returns the full candidate list with:
      - loaded files marked  hot=True  and populated with "content"/"tokens"
      - unloaded files kept  hot=False (no content key added)
    """
    budget_limit = int(TOKEN_BUDGET * threshold)
    used         = already_used
    result       = []

    for meta in candidates:
        headroom = budget_limit - used
        # Peek at size without reading: use byte count as a cheap proxy
        estimated = meta["bytes"] // 4
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


# ── Demo: load sample_project with a tiny artificial budget ─────────────────
# Set threshold so roughly the first half of files (by size) fit.
candidates   = glob_files("*.py", repo_root="sample_project")
total_tokens = sum(c["bytes"] // 4 for c in candidates)
# Target: load ~40% of the total so some files are hot, some cold
TINY_BUDGET  = (total_tokens * 0.40) / TOKEN_BUDGET
loaded_files = budget_load(candidates, repo_root="sample_project", threshold = TINY_BUDGET)

hot  = [f for f in loaded_files if f["hot"]]
cold = [f for f in loaded_files if not f["hot"]]

show_rule("budget_load() result")
show_panel(
    query        = "How does the project handle nested JSON objects?",
    token_used   = sum(f.get("tokens", 0) for f in hot),
    files        = [{"path": f["path"], "hot": f["hot"]} for f in loaded_files],
    strategy     = "glob + JIT",
    prompt_size  = sum(f.get("tokens", 0) for f in hot),
)
print(f"\nHOT ({len(hot)} files, {sum(f['tokens'] for f in hot):,} tokens):")
for f in hot:
    print(f"  ✓ {f['path']}")
print(f"\nCOLD ({len(cold)} files — found but not loaded):")
for f in cold:
    print(f"  ✗ {f['path']}")
```


<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">budget_load() result</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>




<div style="border:1px solid #d0d7de;border-radius:8px;padding:14px 18px;
            margin:8px 0;background:#f6f8fa;font-family:monospace;font-size:0.88rem">
  <div style="font-weight:bold;color:#0969da;margin-bottom:10px">
    Pocket Agent&nbsp;&nbsp;·&nbsp;&nbsp;<em style="font-weight:normal;color:#57606a">How does the project handle nested JSON objects?</em>
  </div>
  <table style="border-collapse:collapse;width:100%">
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:middle;white-space:nowrap">Token Budget</td>
      <td style="vertical-align:middle">
        <div style="display:inline-flex;align-items:center;gap:10px">
          <div style="width:200px;height:10px;border-radius:5px;background:#e0e0e0;overflow:hidden">
            <div style="width:0px;height:100%;background:#2da44e;transition:width 0.3s"></div>
          </div>
          <span style="color:#57606a;font-size:0.82rem">596&nbsp;/&nbsp;262,144 tokens</span>
        </div>
      </td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:top">Retrieved</td>
      <td style="vertical-align:top"><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">main.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">tests/test_formatter.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">tests/test_formatter_gen.py</code></div><div style="margin:2px 0"><span style="background:#0969da;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">COLD</span>&nbsp;&nbsp;<code style="font-size:0.85rem">tests/test_parser.py</code></div><div style="margin:2px 0"><span style="background:#0969da;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">COLD</span>&nbsp;&nbsp;<code style="font-size:0.85rem">utils/formatter.py</code></div><div style="margin:2px 0"><span style="background:#0969da;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">COLD</span>&nbsp;&nbsp;<code style="font-size:0.85rem">utils/parser.py</code></div><div style="margin:2px 0"><span style="background:#0969da;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">COLD</span>&nbsp;&nbsp;<code style="font-size:0.85rem">utils/validator.py</code></div></td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Strategy</td>
      <td style="color:#8250df;font-weight:bold">glob + JIT</td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Prompt&nbsp;size</td>
      <td style="color:#57606a">596 tokens</td>
    </tr>
  </table>
</div>


    
    HOT (3 files, 596 tokens):
      ✓ main.py
      ✓ tests/test_formatter.py
      ✓ tests/test_formatter_gen.py
    
    COLD (4 files — found but not loaded):
      ✗ tests/test_parser.py
      ✗ utils/formatter.py
      ✗ utils/parser.py
      ✗ utils/validator.py


### 5.5 Try it end-to-end

Ask a question about the sample project.
The agent globs for Python files, JIT-loads what fits in the budget,
and sends the HOT files plus the manifest to Ollama.


```python
query      = "How does the parser handle a JSON array of plain numbers like [1, 2, 3]?"
repo       = "sample_project"

# Phase 1: glob — free, no tokens spent
candidates   = glob_files("*.py", repo_root=repo)

# Phase 2: budget-aware JIT load
manifest     = load_manifest(repo)
manifest_tok = manifest["tokens"]
loaded_files = budget_load(candidates, already_used=manifest_tok, repo_root=repo)

hot_files    = [f for f in loaded_files if f["hot"]]
prompt_size  = manifest_tok + sum(f["tokens"] for f in hot_files) + count_tokens(query)

show_rule("Chapter 5 — before LLM call")

reply, tokens_used = ask_with_manifest(query, repo_root=repo, files=hot_files)

show_rule("After call")
show_panel(
    query       = query,
    token_used  = tokens_used,
    files       = [{"path": f["path"], "hot": f["hot"]} for f in loaded_files],
    strategy    = "glob + JIT",
    prompt_size = prompt_size,
)
show_reply(reply)

```


<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">Chapter 5 — before LLM call</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>




<div style="border:1px solid #d0d7de;border-radius:8px;padding:14px 18px;
            margin:8px 0;background:#f6f8fa;font-family:monospace;font-size:0.88rem">
  <div style="font-weight:bold;color:#0969da;margin-bottom:10px">
    Pocket Agent&nbsp;&nbsp;·&nbsp;&nbsp;<em style="font-weight:normal;color:#57606a">How does the parser handle a JSON array of plain numbers like [1, 2, 3]?</em>
  </div>
  <table style="border-collapse:collapse;width:100%">
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:middle;white-space:nowrap">Token Budget</td>
      <td style="vertical-align:middle">
        <div style="display:inline-flex;align-items:center;gap:10px">
          <div style="width:200px;height:10px;border-radius:5px;background:#e0e0e0;overflow:hidden">
            <div style="width:1px;height:100%;background:#2da44e;transition:width 0.3s"></div>
          </div>
          <span style="color:#57606a;font-size:0.82rem">1,751&nbsp;/&nbsp;262,144 tokens</span>
        </div>
      </td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:top">Retrieved</td>
      <td style="vertical-align:top"><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">main.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">tests/test_formatter.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">tests/test_formatter_gen.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">tests/test_parser.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">utils/formatter.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">utils/parser.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">utils/validator.py</code></div></td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Strategy</td>
      <td style="color:#8250df;font-weight:bold">manifest</td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Prompt&nbsp;size</td>
      <td style="color:#57606a">1,751 tokens</td>
    </tr>
  </table>
</div>



<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">After call</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>




<div style="border:1px solid #d0d7de;border-radius:8px;padding:14px 18px;
            margin:8px 0;background:#f6f8fa;font-family:monospace;font-size:0.88rem">
  <div style="font-weight:bold;color:#0969da;margin-bottom:10px">
    Pocket Agent&nbsp;&nbsp;·&nbsp;&nbsp;<em style="font-weight:normal;color:#57606a">How does the parser handle a JSON array of plain numbers like [1, 2, 3]?</em>
  </div>
  <table style="border-collapse:collapse;width:100%">
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:middle;white-space:nowrap">Token Budget</td>
      <td style="vertical-align:middle">
        <div style="display:inline-flex;align-items:center;gap:10px">
          <div style="width:200px;height:10px;border-radius:5px;background:#e0e0e0;overflow:hidden">
            <div style="width:1px;height:100%;background:#2da44e;transition:width 0.3s"></div>
          </div>
          <span style="color:#57606a;font-size:0.82rem">2,191&nbsp;/&nbsp;262,144 tokens</span>
        </div>
      </td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:top">Retrieved</td>
      <td style="vertical-align:top"><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">main.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">tests/test_formatter.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">tests/test_formatter_gen.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">tests/test_parser.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">utils/formatter.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">utils/parser.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">utils/validator.py</code></div></td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Strategy</td>
      <td style="color:#8250df;font-weight:bold">glob + JIT</td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Prompt&nbsp;size</td>
      <td style="color:#57606a">1,680 tokens</td>
    </tr>
  </table>
</div>



The parser handles a JSON array of plain numbers like `[1, 2, 3]` through the `parse_json` function in `utils/parser.py`.

Here's what happens step by step:

1. **JSON parsing**: `json.loads(raw)` converts the string `'[1, 2, 3]'` into a Python list: `[1, 2, 3]`

2. **List type detection**: Since `data` is a list, the parser enters the first `if isinstance(data, list):` branch

3. **Scalar handling**: For each item in the list, the parser checks if it's a dict. Since numbers like `1`, `2`, `3` are not dicts, it wraps each scalar in a dictionary with the key `"value"`:
   - `1` becomes `{"value": 1}`
   - `2` becomes `{"value": 2}`
   - `3` becomes `{"value": 3}`

4. **Result**: The function returns `[{"value": 1}, {"value": 2}, {"value": 3}]`

This behavior is confirmed by the unit test `test_list_of_scalars` in `tests/test_parser.py`:
```python
def test_list_of_scalars(self):
    result = parse_json('[1, 2, 3]')
    assert result == [{"value": 1}, {"value": 2}, {"value": 3}]
```

So when the JSON-to-CSV converter processes `[1, 2, 3]`, it will convert it to a CSV with one column named `"value"` and three rows containing `1`, `2`, and `3` respectively.

The CSV output would look like:
```
value
1
2
3
```


---
## Chapter 6 — Fuzzy Scoring

**Goal:** rank retrieved files before loading them, so budget_load() always drops the *least* relevant files when it runs out of room.

Chapter 5's glob returns files in alphabetical order.
If the budget fills up, it drops whatever comes last alphabetically — which may be the most relevant file.
Fuzzy scoring fixes this by sorting candidates by relevance *before* the JIT load loop.

Scoring works entirely on **file paths** — no content is read.
That keeps it free (no tokens, no disk I/O beyond what glob already did).

**Signals used:**
| Signal | Weight |
|--------|--------|
| Exact query word in filename stem | high |
| Fuzzy match between query word and filename stem | medium |
| Query word appears in a directory component | low |
| File is a test file, query doesn't mention tests | penalty ×1.5 |

**You will:**
- Write `tokenize_query()` — normalise a query into scoreable terms
- Write `score_file()` — score one file against the term list
- Write `rank_files()` — sort a candidate list by score descending
- Replace the alphabetical glob order with ranked order and see the panel change

### 6.1 `tokenize_query()` — extract scoreable terms

We strip stop words and short tokens so scores aren't diluted by
words like "the", "a", "how", "does".


```python
import difflib

_STOP_WORDS = {
    "a", "an", "the", "is", "in", "it", "of", "to", "do", "does",
    "how", "what", "why", "when", "where", "which", "who", "for",
    "and", "or", "but", "not", "with", "this", "that", "are", "was",
    "i", "me", "my", "we", "our", "you", "your",
}


def tokenize_query(query: str) -> list[str]:
    """
    Lowercase, split on non-alphanumeric chars, remove stop words and
    tokens shorter than 3 characters.

    "How does the formatter handle missing fields?"
    → ["formatter", "handle", "missing", "fields"]
    """
    words = re.split(r"[^a-zA-Z0-9]+", query.lower())
    return [w for w in words if len(w) >= 3 and w not in _STOP_WORDS]


# Quick check
print(tokenize_query("How does the formatter handle missing fields?"))
print(tokenize_query("Where is the CSV delimiter configured?"))
```

    ['formatter', 'handle', 'missing', 'fields']
    ['csv', 'delimiter', 'configured']


### 6.2 `score_file()` — score one file against a query

`difflib.SequenceMatcher` gives us a similarity ratio between 0 and 1
with no extra dependencies — it's in the Python standard library.


```python
def score_file(meta: dict, query_terms: list[str]) -> float:
    """
    Score *meta* (a glob result dict) against *query_terms*.

    Higher = more relevant.  Returns 1.0 for an empty term list.
    No file content is read — scoring is path-only.
    """
    # Match weights — stem matters more than directory
    W_STEM_EXACT   = 4.0
    W_STEM_SUBSTR  = 2.5
    W_STEM_FUZZY   = 2.0  # multiplied by SequenceMatcher ratio (0–1)
    W_DIR_EXACT    = 2.0
    W_DIR_SUBSTR   = 1.5
    W_DIR_FUZZY    = 1.3  # multiplied by SequenceMatcher ratio (0–1)
    W_TEST_PENALTY = 0.5  # down-rank test files when query isn't about tests

    if not query_terms:
        return 1.0

    path  = meta["path"].lower()
    parts = Path(path).parts          # ("utils", "parser.py")
    stem  = Path(path).stem           # "parser"
    dirs  = parts[:-1]                # ("utils",)

    score = 1.0
    for term in query_terms:
        # ── Filename stem ────────────────────────────────────────────
        if term == stem:
            score += W_STEM_EXACT
        elif term in stem or stem in term:
            score += W_STEM_SUBSTR
        else:
            score += difflib.SequenceMatcher(None, term, stem).ratio() * W_STEM_FUZZY

        # ── Directory components ──────────────────────────────────────
        for d in dirs:
            if term == d:
                score += W_DIR_EXACT
            elif term in d:
                score += W_DIR_SUBSTR
            else:
                score += difflib.SequenceMatcher(None, term, d).ratio() * W_DIR_FUZZY

    # ── Test-file penalty ─────────────────────────────────────────────
    if "test" in path and "test" not in query_terms:
        score *= W_TEST_PENALTY

    return round(score, 4)


# ── Sanity check ─────────────────────────────────────────────────────────────
_candidates = glob_files("*.py", repo_root="sample_project")
_terms      = tokenize_query("How does the formatter handle missing fields?")

for f in _candidates:
    print(f"{score_file(f, _terms):.4f}  {f['path']}")
```

    3.9063  main.py
    3.0022  tests/test_formatter.py
    3.0003  tests/test_formatter_gen.py
    2.2497  tests/test_parser.py
    7.8812  utils/formatter.py
    4.6722  utils/parser.py
    4.3256  utils/validator.py


### 6.3 `rank_files()` — sort the candidate list

A thin wrapper that applies `score_file()` to every candidate and
returns them sorted highest-score-first.
Files with a score of 0 are kept — they're still candidates,
just lowest priority.


```python
def rank_files(candidates: list[dict], query: str) -> list[dict]:
    """
    Return *candidates* sorted by relevance to *query*, highest first.
    Adds a "score" key to each dict.
    """
    terms = tokenize_query(query)
    scored = [
        {**c, "score": score_file(c, terms)}
        for c in candidates
    ]
    return sorted(scored, key=lambda x: x["score"], reverse=True)


# ── Show ranked order vs alphabetical ────────────────────────────────────────
query      = "How does the formatter handle missing fields?"
candidates = glob_files("*.py", repo_root="sample_project")
ranked     = rank_files(candidates, query)

rows = ""
for i, f in enumerate(ranked, 1):
    color = "#2da44e" if i == 1 else ("#bf8700" if i <= 3 else "#8c959f")
    rows += (
        f'<tr><td style="text-align:right;padding:3px 10px;color:#8c959f">{i}</td>'
        f'<td style="text-align:right;padding:3px 10px;color:{color};font-weight:bold">{f["score"]:.4f}</td>'
        f'<td style="padding:3px 10px;font-family:monospace">{f["path"]}</td></tr>'
    )

display(HTML(f"""
<div style="font-size:0.88rem">
  <b>Ranked candidates</b>
  <table style="border-collapse:collapse;margin-top:6px">
    <tr style="border-bottom:1px solid #d0d7de;color:#57606a">
      <th style="text-align:right;padding:3px 10px">Rank</th>
      <th style="text-align:right;padding:3px 10px">Score</th>
      <th style="text-align:left;padding:3px 10px">File</th>
    </tr>
    {rows}
  </table>
</div>
"""))

```



<div style="font-size:0.88rem">
  <b>Ranked candidates</b>
  <table style="border-collapse:collapse;margin-top:6px">
    <tr style="border-bottom:1px solid #d0d7de;color:#57606a">
      <th style="text-align:right;padding:3px 10px">Rank</th>
      <th style="text-align:right;padding:3px 10px">Score</th>
      <th style="text-align:left;padding:3px 10px">File</th>
    </tr>
    <tr><td style="text-align:right;padding:3px 10px;color:#8c959f">1</td><td style="text-align:right;padding:3px 10px;color:#2da44e;font-weight:bold">7.8812</td><td style="padding:3px 10px;font-family:monospace">utils/formatter.py</td></tr><tr><td style="text-align:right;padding:3px 10px;color:#8c959f">2</td><td style="text-align:right;padding:3px 10px;color:#bf8700;font-weight:bold">4.6722</td><td style="padding:3px 10px;font-family:monospace">utils/parser.py</td></tr><tr><td style="text-align:right;padding:3px 10px;color:#8c959f">3</td><td style="text-align:right;padding:3px 10px;color:#bf8700;font-weight:bold">4.3256</td><td style="padding:3px 10px;font-family:monospace">utils/validator.py</td></tr><tr><td style="text-align:right;padding:3px 10px;color:#8c959f">4</td><td style="text-align:right;padding:3px 10px;color:#8c959f;font-weight:bold">3.9063</td><td style="padding:3px 10px;font-family:monospace">main.py</td></tr><tr><td style="text-align:right;padding:3px 10px;color:#8c959f">5</td><td style="text-align:right;padding:3px 10px;color:#8c959f;font-weight:bold">3.0022</td><td style="padding:3px 10px;font-family:monospace">tests/test_formatter.py</td></tr><tr><td style="text-align:right;padding:3px 10px;color:#8c959f">6</td><td style="text-align:right;padding:3px 10px;color:#8c959f;font-weight:bold">3.0003</td><td style="padding:3px 10px;font-family:monospace">tests/test_formatter_gen.py</td></tr><tr><td style="text-align:right;padding:3px 10px;color:#8c959f">7</td><td style="text-align:right;padding:3px 10px;color:#8c959f;font-weight:bold">2.2497</td><td style="padding:3px 10px;font-family:monospace">tests/test_parser.py</td></tr>
  </table>
</div>



### 6.4 Try it — ranked glob + JIT

Same query as Chapter 5, but now candidates are ranked before budget_load()
sees them. The first file that turns COLD will be the least relevant, not a
random alphabetical casualty.


```python
query = "How does the formatter handle missing fields?"
repo  = "sample_project"

# Phase 1: glob
candidates = glob_files("*.py", repo_root=repo)

# Phase 2: rank  ← new
ranked = rank_files(candidates, query)

# Phase 3: budget-aware JIT load (now works on ranked order)
manifest     = load_manifest(repo)
manifest_tok = manifest["tokens"]
loaded_files = budget_load(ranked, already_used=manifest_tok, repo_root=repo)

hot_files   = [f for f in loaded_files if f["hot"]]
prompt_size = manifest_tok + sum(f["tokens"] for f in hot_files) + count_tokens(query)

show_rule("Chapter 6 — fuzzy-ranked before LLM call")

reply, tokens_used = ask_with_manifest(query, repo_root=repo, files=hot_files)

show_rule("After call")
show_panel(
    query       = query,
    token_used  = tokens_used,
    files       = [{"path": f["path"], "hot": f["hot"]} for f in loaded_files],
    strategy    = "glob + fuzzy + JIT",
    prompt_size = prompt_size,
)
show_reply(reply)

```


<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">Chapter 6 — fuzzy-ranked before LLM call</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>




<div style="border:1px solid #d0d7de;border-radius:8px;padding:14px 18px;
            margin:8px 0;background:#f6f8fa;font-family:monospace;font-size:0.88rem">
  <div style="font-weight:bold;color:#0969da;margin-bottom:10px">
    Pocket Agent&nbsp;&nbsp;·&nbsp;&nbsp;<em style="font-weight:normal;color:#57606a">How does the formatter handle missing fields?</em>
  </div>
  <table style="border-collapse:collapse;width:100%">
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:middle;white-space:nowrap">Token Budget</td>
      <td style="vertical-align:middle">
        <div style="display:inline-flex;align-items:center;gap:10px">
          <div style="width:200px;height:10px;border-radius:5px;background:#e0e0e0;overflow:hidden">
            <div style="width:1px;height:100%;background:#2da44e;transition:width 0.3s"></div>
          </div>
          <span style="color:#57606a;font-size:0.82rem">1,744&nbsp;/&nbsp;262,144 tokens</span>
        </div>
      </td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:top">Retrieved</td>
      <td style="vertical-align:top"><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">utils/formatter.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">utils/parser.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">utils/validator.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">main.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">tests/test_formatter.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">tests/test_formatter_gen.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">tests/test_parser.py</code></div></td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Strategy</td>
      <td style="color:#8250df;font-weight:bold">manifest</td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Prompt&nbsp;size</td>
      <td style="color:#57606a">1,744 tokens</td>
    </tr>
  </table>
</div>



<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">After call</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>




<div style="border:1px solid #d0d7de;border-radius:8px;padding:14px 18px;
            margin:8px 0;background:#f6f8fa;font-family:monospace;font-size:0.88rem">
  <div style="font-weight:bold;color:#0969da;margin-bottom:10px">
    Pocket Agent&nbsp;&nbsp;·&nbsp;&nbsp;<em style="font-weight:normal;color:#57606a">How does the formatter handle missing fields?</em>
  </div>
  <table style="border-collapse:collapse;width:100%">
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:middle;white-space:nowrap">Token Budget</td>
      <td style="vertical-align:middle">
        <div style="display:inline-flex;align-items:center;gap:10px">
          <div style="width:200px;height:10px;border-radius:5px;background:#e0e0e0;overflow:hidden">
            <div style="width:1px;height:100%;background:#2da44e;transition:width 0.3s"></div>
          </div>
          <span style="color:#57606a;font-size:0.82rem">2,309&nbsp;/&nbsp;262,144 tokens</span>
        </div>
      </td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:top">Retrieved</td>
      <td style="vertical-align:top"><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">utils/formatter.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">utils/parser.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">utils/validator.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">main.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">tests/test_formatter.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">tests/test_formatter_gen.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">tests/test_parser.py</code></div></td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Strategy</td>
      <td style="color:#8250df;font-weight:bold">glob + fuzzy + JIT</td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Prompt&nbsp;size</td>
      <td style="color:#57606a">1,673 tokens</td>
    </tr>
  </table>
</div>



The formatter (`utils/formatter.py`) handles missing fields by **rendering them as empty strings**.

Here's how it works:

1. **Header Collection**: The formatter first collects all headers from *all* records, preserving the order of first appearance. For example, if one record has `{"a": 1, "b": 2}` and another has `{"a": 3, "c": 4}`, headers will be `["a", "b", "c"]`.

2. **Safe Field Extraction with Fallback**: For each record, it creates a dictionary for `csv.DictWriter` using:
   ```python
   {h: rec.get(h, "") for h in headers}
   ```
   This means:
   - If a header `h` exists in the current record → use its value
   - If `h` is missing → use `""` (empty string) as fallback

3. **Evidence in Tests**:
   - `test_missing_fields_become_empty()` explicitly tests this:  
     Input: `[{"a": 1, "b": 2}, {"a": 3}]`  
     Output CSV row for second record: `"3,"` (missing `b` → empty string)
   - `test_to_csv_missing_values()` confirms:  
     Input: `[{"a": "1", "b": "2"}, {"a": "3", "c": "4"}]`  
     Output: `"1,2,\r\n3,,4\r\n"` → missing `c` in row 1, missing `b` in row 2 both become `""`

4. **Integration with CSV Module**:  
   `extrasaction="ignore"` in `DictWriter` ensures that even if records have *extra* keys (beyond headers), they're safely ignored — but missing keys are the primary concern handled via `.get(..., "")`.

**Visual Example**:
```python
records = [
    {"name": "Alice", "email": "a@x.com"},
    {"name": "Bob"}  # missing 'email'
]
to_csv(records)
# → "name,email\r\nAlice,a@x.com\r\nBob,\r\n"
#                          ^^^^^^
#                         empty string for missing field
```

So: **No field is left unaccounted for** — all expected columns always appear, with missing values as empty strings.


---
## Chapter 7 — Grep

**Goal:** find files by what's *inside* them, not just their name.

Chapters 4–5 rank files by path relevance. That works well when the
query mentions a filename — "how does the *formatter* work?" — but fails
when the relevant file has a generic name.

Consider: *"Where does the code raise a TypeError?"*
No filename contains "typeerror". Fuzzy scoring gives every file a near-zero
path score. But `validator.py` *contains* `raise TypeError(...)` — and grep
finds it instantly.

Grep is the third retrieval strategy. It searches file content for
patterns derived from the query, counts matches per file, and uses that
count as an additional relevance signal on top of the fuzzy path score.

**You will:**
- Write `grep_repo()` — search file contents, return matches with excerpts
- Write `query_to_patterns()` — turn query terms into search patterns
- Write `grep_rank()` — combine grep hit count with fuzzy path score
- See grep surface `validator.py` for a query that fuzzy scoring misses entirely

### 7.1 `grep_repo()` — search file contents

For each file, we search for a compiled regex pattern and collect:
- the number of matching lines (used for ranking)
- up to `context_lines` surrounding each match (used in the prompt as an excerpt)

Returning excerpts rather than full file content is a key budget trick:
if grep finds 2 matching lines in a 300-line file, we can show just those
2 lines plus context instead of loading all 300 lines HOT.


```python
def grep_repo(
    pattern:       str,
    repo_root:     str   = REPO_ROOT,
    extensions:    set   = CODE_EXTENSIONS,
    context_lines: int   = 2,
    max_matches:   int   = 5,
) -> list[dict]:
    """
    Search every matching file under *repo_root* for *pattern* (regex).

    Returns a list of dicts — one per file that had at least one match:
        {
          "path":      str,         # relative to repo_root
          "hit_count": int,         # number of matching lines
          "excerpt":   str,         # up to max_matches hits with context
          "bytes":     int,
          "hot":       False,       # budget_load() will flip this
          "score":     1.0,         # grep_rank() will fill this
        }
    Sorted by hit_count descending.
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

            # Find matching line indices
            hit_indices = [i for i, ln in enumerate(lines) if regex.search(ln)]
            if not hit_indices:
                continue

            # Build excerpt: up to max_matches hits with context
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

            excerpt_text = "\n\n".join(excerpts)

            results.append({
                "path":      str(full.relative_to(root)),
                "hit_count": len(hit_indices),
                "excerpt":   excerpt_text,
                "bytes":     full.stat().st_size,
                "hot":       False,
                "score":     1.0,
            })

    return sorted(results, key=lambda x: x["hit_count"], reverse=True)


# ── Demo: search for TypeError ───────────────────────────────────────────────
hits = grep_repo("TypeError", repo_root="sample_project")
for h in hits:
    print(f"{h['hit_count']} hit(s)  {h['path']}")
    print(h["excerpt"])
    print()
```

    3 hit(s)  utils/validator.py
         9 │     field names to expected types: "string", "number", "boolean").
        10 │ 
    →   11 │     Raises TypeError on the first violation found.
        12 │     """
        13 │     schema: dict[str, str] = json.loads(schema_raw)
    
        17 │         for field, expected_type_name in schema.items():
        18 │             if field not in record:
    →   19 │                 raise TypeError(
        20 │                     f"Record {i}: missing required field '{field}'"
        21 │                 )
    
        22 │             expected = type_map.get(expected_type_name)
        23 │             if expected and not isinstance(record[field], expected):
    →   24 │                 raise TypeError(
        25 │                     f"Record {i}: field '{field}' expected {expected_type_name}, "
        26 │                     f"got {type(record[field]).__name__}"
    


### 7.2 `query_to_patterns()` — derive search patterns from a query

We turn the query terms into a single alternation regex:
`formatter|missing|fields`

This is a liberal match — any file mentioning *any* term is a candidate.
We'll use hit count to separate strong matches from weak ones.


```python
def query_to_patterns(query: str) -> str:
    """
    Convert a natural-language query into a single regex pattern
    suitable for grep_repo().

    "Where does the code raise a TypeError?"
    → "typeerror|raise|code"   (terms ≥ 4 chars, lowercased, joined with |)

    Returns an empty string if no usable terms are found.
    """
    terms = [t for t in tokenize_query(query) if len(t) >= 4]
    if not terms:
        return ""
    return "|".join(re.escape(t) for t in terms)


# Check a few examples
for q in [
    "Where does the code raise a TypeError?",
    "How does the formatter handle missing fields?",
    "What is the CSV delimiter default value?",
]:
    print(f"  {q!r}\n  → {query_to_patterns(q)!r}\n")
```

      'Where does the code raise a TypeError?'
      → 'code|raise|typeerror'
    
      'How does the formatter handle missing fields?'
      → 'formatter|handle|missing|fields'
    
      'What is the CSV delimiter default value?'
      → 'delimiter|default|value'
    


### 7.3 `grep_rank()` — combine grep hits with fuzzy path score

A file can score high two ways:
- Many grep hits → strong content signal
- High fuzzy path score → strong name signal

`grep_rank()` normalises both signals to [0, 1] and adds them,
so a file with a relevant name *and* relevant content beats one with only one signal.


```python
def grep_rank(
    query:     str,
    repo_root: str = REPO_ROOT,
) -> list[dict]:
    """
    Full grep-based retrieval pipeline for *query*:
    1. Derive a regex pattern from the query terms
    2. Grep every source file for that pattern
    3. Score each hit file with both grep hit_count and fuzzy path score
    4. Normalise both signals to [0, 1] and sum them
    5. Return sorted by combined score descending
    Files with zero grep hits are excluded entirely.
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
        norm_grep = h["hit_count"]    / max_hits
        norm_path = score_file(h, terms) / max_path_score
        combined  = round(norm_grep + norm_path, 4)
        ranked.append({**h, "score": combined})
    return sorted(ranked, key=lambda x: x["score"], reverse=True)


# ── Demo ──────────────────────────────────────────────────────────────────────
query  = "Where does the code raise a TypeError?"
ranked = grep_rank(query, repo_root="sample_project")

rows = ""
for f in ranked:
    excerpt = (f["excerpt"].splitlines()[0] if f["excerpt"] else "").replace("<","&lt;")
    rows += (
        f'<tr>'
        f'<td style="text-align:right;padding:3px 10px;color:#2da44e;font-weight:bold">{f["score"]}</td>'
        f'<td style="text-align:right;padding:3px 10px">{f["hit_count"]}</td>'
        f'<td style="padding:3px 10px;font-family:monospace">{f["path"]}</td>'
        f'<td style="padding:3px 10px;font-family:monospace;color:#57606a;font-size:0.82rem">{excerpt}</td>'
        f'</tr>'
    )

display(HTML(f"""
<div style="font-size:0.88rem">
  <b>grep_rank(): {query!r}</b>
  <table style="border-collapse:collapse;margin-top:6px">
    <tr style="border-bottom:1px solid #d0d7de;color:#57606a">
      <th style="text-align:right;padding:3px 10px">Score</th>
      <th style="text-align:right;padding:3px 10px">Hits</th>
      <th style="text-align:left;padding:3px 10px">File</th>
      <th style="text-align:left;padding:3px 10px">First match</th>
    </tr>
    {rows}
  </table>
</div>
"""))

fuzzy_only = rank_files(glob_files("*.py", repo_root="sample_project"), query)
print("\nFuzzy path scores for same query (for comparison):")
for f in fuzzy_only:
    print(f"  {f['score']:.4f}  {f['path']}")

```



<div style="font-size:0.88rem">
  <b>grep_rank(): 'Where does the code raise a TypeError?'</b>
  <table style="border-collapse:collapse;margin-top:6px">
    <tr style="border-bottom:1px solid #d0d7de;color:#57606a">
      <th style="text-align:right;padding:3px 10px">Score</th>
      <th style="text-align:right;padding:3px 10px">Hits</th>
      <th style="text-align:left;padding:3px 10px">File</th>
      <th style="text-align:left;padding:3px 10px">First match</th>
    </tr>
    <tr><td style="text-align:right;padding:3px 10px;color:#2da44e;font-weight:bold">2.0</td><td style="text-align:right;padding:3px 10px">4</td><td style="padding:3px 10px;font-family:monospace">utils/parser.py</td><td style="padding:3px 10px;font-family:monospace;color:#57606a;font-size:0.82rem">    11 │     - a list of scalars   → each scalar wrapped as {"value": scalar}</td></tr><tr><td style="text-align:right;padding:3px 10px;color:#2da44e;font-weight:bold">1.4921</td><td style="text-align:right;padding:3px 10px">3</td><td style="padding:3px 10px;font-family:monospace">utils/validator.py</td><td style="padding:3px 10px;font-family:monospace;color:#57606a;font-size:0.82rem">     9 │     field names to expected types: "string", "number", "boolean").</td></tr><tr><td style="text-align:right;padding:3px 10px;color:#2da44e;font-weight:bold">1.4675</td><td style="text-align:right;padding:3px 10px">4</td><td style="padding:3px 10px;font-family:monospace">tests/test_parser.py</td><td style="padding:3px 10px;font-family:monospace;color:#57606a;font-size:0.82rem">    18 │         assert result == [{"value": 1}, {"value": 2}, {"value": 3}]</td></tr>
  </table>
</div>



    
    Fuzzy path scores for same query (for comparison):
      3.9966  utils/parser.py
      3.8449  utils/formatter.py
      2.9658  utils/validator.py
      1.8889  main.py
      1.8685  tests/test_parser.py
      1.8460  tests/test_formatter.py
      1.6991  tests/test_formatter_gen.py


### 7.4 Try it — grep-guided query with excerpt injection

One more trick: instead of loading the *full* file HOT, we can inject
just the grep **excerpt** for files that matched well but are expensive
to load in full. This stretches the budget further.

For this demo we load full content (the sample files are small), but
the `excerpt` field is there and Chapter 8 will start using it during compaction.


```python
query = "Where does the code raise a TypeError?"
repo  = "sample_project"

# Grep-ranked candidates (includes excerpt, excludes files with zero hits)
candidates = grep_rank(query, repo_root=repo)

# Fall back to fuzzy glob for files with zero grep hits
all_paths    = {f["path"] for f in candidates}
glob_extras  = [f for f in rank_files(glob_files("*.py", repo_root=repo), query)
                if f["path"] not in all_paths]
candidates   = candidates + glob_extras   # grep hits first, fuzzy extras after

# Budget-aware JIT load
manifest     = load_manifest(repo)
manifest_tok = manifest["tokens"]
loaded_files = budget_load(candidates, already_used=manifest_tok, repo_root=repo)

hot_files   = [f for f in loaded_files if f["hot"]]
prompt_size = manifest_tok + sum(f["tokens"] for f in hot_files) + count_tokens(query)

show_rule("Chapter 7 — grep-ranked before LLM call")
# show_panel(
#     query        = query,
#     token_used   = prompt_size,
#     files        = [{"path": f["path"], "hot": f["hot"]} for f in loaded_files],
#     strategy     = "grep + fuzzy + JIT",
#     prompt_size  = prompt_size,
# )

reply, tokens_used = ask_with_manifest(query, repo_root=repo, files=hot_files)

show_rule("After call")
show_panel(
    query       = query,
    token_used  = tokens_used,
    files       = [{"path": f["path"], "hot": f["hot"]} for f in loaded_files],
    strategy    = "grep + fuzzy + JIT",
    prompt_size = prompt_size,
)
show_reply(reply)

```


<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">Chapter 7 — grep-ranked before LLM call</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>




<div style="border:1px solid #d0d7de;border-radius:8px;padding:14px 18px;
            margin:8px 0;background:#f6f8fa;font-family:monospace;font-size:0.88rem">
  <div style="font-weight:bold;color:#0969da;margin-bottom:10px">
    Pocket Agent&nbsp;&nbsp;·&nbsp;&nbsp;<em style="font-weight:normal;color:#57606a">Where does the code raise a TypeError?</em>
  </div>
  <table style="border-collapse:collapse;width:100%">
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:middle;white-space:nowrap">Token Budget</td>
      <td style="vertical-align:middle">
        <div style="display:inline-flex;align-items:center;gap:10px">
          <div style="width:200px;height:10px;border-radius:5px;background:#e0e0e0;overflow:hidden">
            <div style="width:1px;height:100%;background:#2da44e;transition:width 0.3s"></div>
          </div>
          <span style="color:#57606a;font-size:0.82rem">1,742&nbsp;/&nbsp;262,144 tokens</span>
        </div>
      </td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:top">Retrieved</td>
      <td style="vertical-align:top"><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">utils/parser.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">utils/validator.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">tests/test_parser.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">utils/formatter.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">main.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">tests/test_formatter.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">tests/test_formatter_gen.py</code></div></td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Strategy</td>
      <td style="color:#8250df;font-weight:bold">manifest</td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Prompt&nbsp;size</td>
      <td style="color:#57606a">1,742 tokens</td>
    </tr>
  </table>
</div>



<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">After call</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>




<div style="border:1px solid #d0d7de;border-radius:8px;padding:14px 18px;
            margin:8px 0;background:#f6f8fa;font-family:monospace;font-size:0.88rem">
  <div style="font-weight:bold;color:#0969da;margin-bottom:10px">
    Pocket Agent&nbsp;&nbsp;·&nbsp;&nbsp;<em style="font-weight:normal;color:#57606a">Where does the code raise a TypeError?</em>
  </div>
  <table style="border-collapse:collapse;width:100%">
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:middle;white-space:nowrap">Token Budget</td>
      <td style="vertical-align:middle">
        <div style="display:inline-flex;align-items:center;gap:10px">
          <div style="width:200px;height:10px;border-radius:5px;background:#e0e0e0;overflow:hidden">
            <div style="width:3px;height:100%;background:#2da44e;transition:width 0.3s"></div>
          </div>
          <span style="color:#57606a;font-size:0.82rem">4,471&nbsp;/&nbsp;262,144 tokens</span>
        </div>
      </td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:top">Retrieved</td>
      <td style="vertical-align:top"><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">utils/parser.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">utils/validator.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">tests/test_parser.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">utils/formatter.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">main.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">tests/test_formatter.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">tests/test_formatter_gen.py</code></div></td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Strategy</td>
      <td style="color:#8250df;font-weight:bold">grep + fuzzy + JIT</td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Prompt&nbsp;size</td>
      <td style="color:#57606a">1,671 tokens</td>
    </tr>
  </table>
</div>



The code raises a `TypeError` in **`utils/validator.py`**, specifically in the `validate_schema` function.

### Here's the exact line:

```python
if expected and not isinstance(record[field], expected):
```

### Why?

A `TypeError` is raised **if `record[field]` is a dictionary (or other non-primitive type)** and you try to use `isinstance()` with `expected = (int, float)` for `"number"`.

Wait — that shouldn't raise a `TypeError`. In Python, `isinstance(obj, (int, float))` works fine even if `obj` is a dict or any other type.

So why *would* a `TypeError` occur?

Let’s look carefully.

The `type_map` is:

```python
type_map = {"string": str, "number": (int, float), "boolean": bool}
```

And the check is:

```python
if expected and not isinstance(record[field], expected):
```

That’s safe.

BUT — here's the real culprit:

> 🔥 **`isinstance(record[field], expected)` raises `TypeError` if `expected` is not a class or tuple of classes**.

But in our code, `expected` comes from `type_map.get(expected_type_name)`, where `expected_type_name` is a string like `"number"`, and `type_map["number"]` is `(int, float)` — a valid type tuple.

So still safe.

Wait — but what if `expected_type_name` is something **invalid**, like `"object"` or `"array"`? Then:

```python
expected = type_map.get(expected_type_name)  # returns None
if expected and not isinstance(...):          # if expected is None, short-circuits → no problem
```

So no `TypeError` there.

But there’s **one scenario** where a `TypeError` *can* be raised:

### ✅ **Yes! Here:**

```python
for field, expected_type_name in schema.items():
    ...
    expected = type_map.get(expected_type_name)
    if expected and not isinstance(record[field], expected):
        raise TypeError(...)
```

➡️ `record[field]` could be **a list or dict**, and if `expected_type_name` is `"string"` (so `expected = str`), then `isinstance(record[field], str)` is fine.

Still safe.

Wait — what if `schema` contains a **type name not in `type_map`**, like `"array"`?

Then:

```python
expected = type_map.get("array")  # → None
if expected and ...:              # short-circuits — no isinstance → no TypeError
```

Still safe.

So… where is the `TypeError`?

Let’s reexamine the spec:  
> **“Raises TypeError on the first violation found.”**

But `validate_schema` only raises `TypeError` in two cases:

1. Missing field (`field not in record`)
2. Wrong type (`not isinstance(...)`)

The *wrong type* case only raises `TypeError`, but only if the *actual* type check passes — and it shouldn’t raise `TypeError` *unless something’s wrong* with the `isinstance` call itself.

But `isinstance(x, T)` only raises `TypeError` if `T` is not a type or tuple of types.

And `expected` *is* a tuple of types (e.g., `(int, float)`).

So… **where?**

Ah — here’s the subtle bug:

> 🔥 **`isinstance(record[field], expected)` raises `TypeError` if `record[field]` is a class (not an instance), or some exotic non-standard object?**  
→ No — `isinstance` is robust.

Wait — consider this:

What if `record[field]` is a **class object**, like `dict` or `list`?  
Example: `record = {"x": dict}`  
Then `isinstance(dict, (int, float))` → still works: `False`, no `TypeError`.

Still safe.

Let’s look at the actual test usage.

In `tests/test_formatter_gen.py`, `test_to_csv_custom_delimiter`, the value `"95"` is passed — a string — and validated as `"string"` (correct).

But what if a **record has a value that’s a `dict` or `list`**, and schema expects `"string"` or `"number"`?

Example:
```python
records = [{"name": "Alice", "age": {"years": 30}}]
schema_raw = '{"name": "string", "age": "number"}'
```

Then:
```python
expected = (int, float)
isinstance({"years": 30}, (int, float)) → False ✅, no exception
```

Still safe.

So… where is the `TypeError`?

Let’s look again at the error *message* in the prompt:

> **“Where does the code raise a TypeError?”**

Maybe **not** in `validate_schema`, but in **another module**?

Let’s check `utils/parser.py` for `TypeError`:

```python
def parse_json(raw: str) -> list[dict]:
    ...
    if isinstance(data, list):
        return [_flatten(item) if isinstance(item, dict) else {"value": item}
                for item in data]
```

Wait — if `item` is a **dict**, it calls `_flatten(item)`.

What if `_flatten(item)` is called with `item = None`, `item = "string"`, etc.?  
→ No — `if isinstance(item, dict)` protects.

But in `_flatten`:

```python
def _flatten(obj: dict, prefix: str = "", sep: str = ".") -> dict:
    result = {}
    for key, value in obj.items():  # ← obj.items() → TypeError if obj is not dict-like
```

But `_flatten` is only called when `isinstance(item, dict)` — safe.

Now — check `utils/formatter.py`’s `to_csv`:

```python
writer = csv.DictWriter(buf, fieldnames=headers,
                        delimiter=delimiter, extrasaction="ignore")
...
for rec in records:
    writer.writerow({h: rec.get(h, "") for h in headers})
```

What if `rec.get(h, "")` returns a **dict or list**, and `csv.DictWriter.writerow()` tries to write it?

→ `csv.DictWriter.writerow()` expects *all* values to be strings or coercible to strings (e.g., `str()`-able).

If a value is a **non-string, non-number** type (e.g., `dict`), `writer.writerow()` will call `str()` on it — that works.

But wait — `DictWriter.writerow()` uses `self.writer.writerow(rowdict)` (which is `csv.writer.writerow()`).

From Python docs:

> `writerow()` raises `TypeError` if `rowdict` contains non-string keys, but values can be any type — `csv` will convert them using `str()`.

So no `TypeError`.

Let’s check `main.py`:

```python
records = parse_json(raw)
validate_schema(records, schema)
```

Ah — what if `validate_schema` is called **after** parsing, and some `record` contains a `dict` as a value?

Example:
```python
parse_json('{"a": {"b": 1}}') → [{'a.b': 1}] → flat, so safe.

But what if user passes:
{"x": {"y": 2}} → parsed to {"x.y": 2}.

So still flat.

Wait — what if `_flatten` is broken? No — `_flatten` expects `dict`, and only flattens nested dicts.

But in `parse_json`, for list of scalars:

```python
return [_flatten(item) if isinstance(item, dict) else {"value": item} ...]
```

So if `item` is **not** a dict, it’s wrapped as `{"value": item}`.

So values can be scalars (int, str, bool) — fine.

But what if `item` is a **list**, like:

```python
parse_json('[{"a": [1,2,3]}]')
```

→ `{"a": [1,2,3]}` is a dict, so `_flatten({"a": [1,2,3]})` → `{"a": [1,2,3]}` (list not dict → not recursed).

So `{"a": [1,2,3]}` — a list as value.

Then `validate_schema` is called with schema like:

```json
{ "a": "string" }
```

Then:

```python
isinstance([1,2,3], str) → False → raises `TypeError(...)`
```

Still not a `TypeError` — just raises the exception with message.

Wait — the question is: **Where does the code raise a TypeError?**

The `validate_schema` function explicitly raises `TypeError` in two cases:

- missing field  
- wrong type

So `TypeError` is **explicitly raised in `utils/validator.py`**, line:

```python
raise TypeError(f"Record {i}: missing required field '{field}'")
```

and

```python
raise TypeError(f"Record {i}: field '{field}' expected {expected_type_name}, got {type(record[field]).__name__}")
```

But neither of those is a *case* of unexpected `TypeError`. The question likely asks: **Where does the code *incorrectly* raise a `TypeError` due to a bug?**

Let’s re-check `utils/parser.py`, line:

```python
return [_flatten(item) if isinstance(item, dict) else {"value": item}
        for item in data]
```

What if `data = [None]`? Then for `item = None`, `{"value": None}` → fine.

What if `data = [()]` (a tuple)? → `{"value": ()}` → fine.

Now — consider this scenario:

In `utils/parser.py`, if the input is:

```json
[{"a": {"b": 1}}, {"x": 2}]
```

→ both are dicts → both passed to `_flatten`.

No issue.

Wait — here's the real bug:

> 🔥 **What if `record[field]` is a `class`, like `str`?**  
`isinstance(str, str)` → `True`  
`isinstance(str, (int, float))` → `False`  
Still no `TypeError`.

Wait — maybe the answer is simpler.

Let’s look at the **actual exception types** raised:

- `parse_json` raises `ValueError` only (on JSON error, unexpected type)
- `_flatten` → no explicit exceptions
- `to_csv` → no explicit `TypeError`
- `validate_schema` → explicitly raises `TypeError`

✅ So the **only place that raises `TypeError`** is:

> ✅ **`utils/validator.py`, lines 27 and 32**

Specifically:

- `raise TypeError(...)` for missing field  
- `raise TypeError(...)` for type mismatch

That’s the answer — `TypeError` is **raised intentionally** in `utils/validator.py`.

But the question is: *“Where does the code raise a TypeError?”*  
Not: *“Where does the code incorrectly raise a TypeError?”*

So the answer is:

> ✅ **In `utils/validator.py`, in the `validate_schema` function — specifically when checking for missing fields or type mismatches.**

But let’s confirm: is there *any other* `TypeError`?

Let’s test a scenario — what if `schema` contains `"number": "number"` → but schema is a JSON string, parsed to `{"age": "number"}`, and `"number"` is a string — fine.

Now — consider this:

What if `expected_type_name` is not `"string"`, `"number"`, or `"boolean"`? Like `"array"` — then `expected = None`, and:

```python
if expected and not isinstance(...):
```

→ short-circuits — no `TypeError`.

What if `expected_type_name` is `None`? Impossible — schema keys and values come from JSON, so always strings.

✅ Final answer:

> **The code raises a `TypeError` in `utils/validator.py`, in the `validate_schema` function, when validating records. Specifically, it raises `TypeError` explicitly in two places:**
> - **When a required field is missing (line 27).**
> - **When a field’s value is not of the expected type (line 32).**


---
## Chapter 8 — Microcompaction

**Goal:** survive long sessions without hitting the context ceiling.

After several turns the conversation grows: loaded file contents, prior replies,
system prompts. Eventually the budget fills up and the model starts forgetting
the beginning of the conversation.

Microcompaction solves this with a simple rule:

> When the running token count exceeds a **compaction threshold**,
> take the coldest HOT files, summarise each one in a single LLM call,
> replace the full content with the summary, and mark them COLD.

The summary is much smaller than the original — typically 10–15 % of the
original token count. Budget is freed; the session continues.

```
Before compaction            After compaction
─────────────────            ────────────────
HOT  utils/parser.py  420t   COLD utils/parser.py  [summary: 45t]
HOT  utils/formatter  380t   HOT  utils/formatter  380t   ← kept
HOT  utils/validator  290t   HOT  utils/validator  290t   ← kept
─────────────────            ────────────────
total: 1090t                 total: 715t   (saved 375t)
```

**You will:**
- Write `eviction_candidates()` — pick which HOT files to evict
- Write `summarise_file()` — compress one file to a short summary via Ollama
- Write `compact()` — orchestrate eviction + summarisation, return updated file list
- Simulate a session running over budget and watch compaction kick in

### 8.1 `eviction_candidates()` — which HOT files to evict first

We evict the files that are least likely to be needed again.
The heuristic: **lowest score first** (from Chapter 6's fuzzy scoring).
Files with no score get evicted before files that matched the current query.

We also always keep at least one HOT file — evicting everything defeats the purpose.


```python
COMPACT_THRESHOLD = 0.80   # trigger compaction when prompt > 80 % of budget
EVICT_TARGET      = 0.55   # compact down to 55 % of budget


def eviction_candidates(
    files:  list[dict],
    query:  str,
    n_keep: int = 1,
) -> tuple[list[dict], list[dict]]:
    """
    Split HOT files into (evict, keep) by relevance score.
    Lowest-scoring HOT files are evicted first; top *n_keep* are retained.
    COLD files are not touched — they're never in the prompt to begin with.

    Returns (to_evict, to_keep) — both lists contain only HOT files.
    """
    hot = [f for f in files if f.get("hot")]

    if len(hot) <= n_keep:
        return [], hot

    terms  = tokenize_query(query)
    scored = sorted(hot, key=lambda f: f.get("score", score_file(f, terms)))

    return scored[:-n_keep], scored[-n_keep:]


# ── Quick check ──────────────────────────────────────────────────────────────
_demo_files = [
    {"path": "utils/parser.py",    "hot": True,  "score": 1.8,  "tokens": 420},
    {"path": "utils/formatter.py", "hot": True,  "score": 3.1,  "tokens": 380},
    {"path": "utils/validator.py", "hot": True,  "score": 1.3,  "tokens": 290},
    {"path": "main.py",            "hot": False, "score": 1.1,  "tokens": 120},
]
evict, keep = eviction_candidates(_demo_files, "how does the formatter work?")
assert [f["path"] for f in evict] == ["utils/validator.py", "utils/parser.py"]
assert [f["path"] for f in keep]  == ["utils/formatter.py"]
print("eviction_candidates ✓")
```

    eviction_candidates ✓


### 8.2 `summarise_file()` — compress one file with the LLM

We ask the model for a terse technical summary in plain prose — no code blocks,
no bullet points. The goal is a 3–5 sentence description that preserves
the key exported names and their purpose, so later queries can still
reference this file even though its full content is no longer in the prompt.


```python
def summarise_file(file_dict: dict) -> dict:
    """
    Ask the LLM to compress *file_dict["content"]* to a short summary.

    Returns a new dict with:
      - "content"  replaced by the summary text
      - "tokens"   updated to the summary's token count
      - "hot"      set to False  (evicted from active prompt)
      - "summary"  set to True   (flag so later code knows this is compressed)
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


# ── Demo: summarise parser.py ────────────────────────────────────────────────
_parser_meta   = glob_files("parser.py", repo_root="sample_project")[0]
_parser_loaded = jit_read(_parser_meta, repo_root="sample_project")

print(f"Before: {_parser_loaded['tokens']} tokens")
_parser_summary = summarise_file(_parser_loaded)
print(f"After : {_parser_summary['tokens']} tokens  "
      f"({_parser_summary['tokens']/_parser_loaded['tokens']*100:.0f}% of original)\n")
print(_parser_summary["content"])
```

    Before: 303 tokens
    After : 145 tokens  (48% of original)
    
    [SUMMARY of utils/parser.py]
    The `parse_json` function takes a JSON string and converts it into a list of flat dictionaries, handling three cases: if the input is already a list of objects, it processes each object; if it’s a single object, it wraps it in a list; and if it’s a list of scalar values, each scalar is wrapped in a dictionary under the key `"value"`. It raises a `ValueError` for invalid JSON or unexpected input types. The `_flatten` helper function recursively flattens nested dictionaries by joining nested keys with a dot separator, turning structures like `{"a": {"b": 1}}` into `{"a.b": 1}`.


### 8.3 `compact()` — orchestrate the full compaction pass

`compact()` is called by the agent whenever the token count crosses
`COMPACT_THRESHOLD`. It loops through eviction candidates, summarises
each one, and returns the updated file list plus a log of what was freed.


```python
def compact(
    files:        list[dict],
    query:        str,
    token_used:   int,
    token_budget: int   = None,
    threshold:    float = COMPACT_THRESHOLD,
    evict_target: float = EVICT_TARGET,
) -> tuple[list[dict], int, list[str]]:
    """
    If *token_used* exceeds threshold × token_budget, evict and summarise
    the least-relevant HOT files until usage drops below evict_target × token_budget.

    token_budget defaults to TOKEN_BUDGET (the global context window).
    Pass a smaller value in demos/tests to trigger compaction on tiny file sets.

    Loop exits when either:
      - token usage drops below evict_target × token_budget, or
      - only n_keep HOT files remain (nothing left to evict)

    Returns:
        (updated_files, new_token_used, log_lines)
    where log_lines is a human-readable list of what was compacted.
    """
    budget    = token_budget if token_budget is not None else TOKEN_BUDGET
    ceiling   = int(budget * threshold)
    target    = int(budget * evict_target)

    if token_used <= ceiling:
        return files, token_used, []   # nothing to do

    log     = [f"Compaction triggered: {token_used:,} / {budget:,} tokens "
               f"({token_used/budget*100:.0f}%)"]
    updated = list(files)
    used    = token_used

    while used > target:
        to_evict, _ = eviction_candidates(updated, query, n_keep=1)
        if not to_evict:
            break

        victim = to_evict[0]
        before = victim.get("tokens", 0)

        summarised = summarise_file(victim)
        after      = summarised["tokens"]
        saved      = before - after

        updated = [summarised if f["path"] == victim["path"] else f
                   for f in updated]
        used   -= saved
        log.append(f"  compacted {victim['path']}: {before}t → {after}t  (saved {saved}t)")

    log.append(f"After compaction: {used:,} / {budget:,} tokens "
               f"({used/budget*100:.0f}%)")
    return updated, used, log
```

### 8.4 Simulation — watch compaction fire

We simulate a session that loads all sample files and artificially inflates
the token count past the compaction threshold, then watch `compact()` log
what it evicts and how much budget it recovers.


```python
query = "How does the formatter handle missing fields?"
repo  = "sample_project"

# Load all Python files HOT
all_files  = glob_files("*.py", repo_root=repo)
loaded     = [jit_read(f, repo_root=repo) for f in all_files]
terms      = tokenize_query(query)
loaded     = [{**f, "score": score_file(f, terms)} for f in loaded]
token_used = sum(f["tokens"] for f in loaded)

# Use a demo budget smaller than the real content so compaction fires on
# actual file summaries — no artificial padding needed.
demo_budget = int(token_used * 0.6)

show_rule("Before compaction")
show_panel(
    query        = query,
    token_used   = token_used,
    token_budget = demo_budget,
    files        = [{"path": f["path"], "hot": f["hot"]} for f in loaded],
    strategy     = "compact",
    prompt_size  = token_used,
)

# Run compaction against the small demo budget
updated, new_used, log = compact(loaded, query, token_used, token_budget=demo_budget)

show_rule("Compaction log")
for line in log:
    print(f"  {line}")

show_rule("After compaction")
show_panel(
    query        = query,
    token_used   = new_used,
    token_budget = demo_budget,
    files        = [{"path": f["path"], "hot": f["hot"]} for f in updated],
    strategy     = "compact",
    prompt_size  = new_used,
)
```


<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">Before compaction</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>




<div style="border:1px solid #d0d7de;border-radius:8px;padding:14px 18px;
            margin:8px 0;background:#f6f8fa;font-family:monospace;font-size:0.88rem">
  <div style="font-weight:bold;color:#0969da;margin-bottom:10px">
    Pocket Agent&nbsp;&nbsp;·&nbsp;&nbsp;<em style="font-weight:normal;color:#57606a">How does the formatter handle missing fields?</em>
  </div>
  <table style="border-collapse:collapse;width:100%">
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:middle;white-space:nowrap">Token Budget</td>
      <td style="vertical-align:middle">
        <div style="display:inline-flex;align-items:center;gap:10px">
          <div style="width:200px;height:10px;border-radius:5px;background:#e0e0e0;overflow:hidden">
            <div style="width:200px;height:100%;background:#cf222e;transition:width 0.3s"></div>
          </div>
          <span style="color:#57606a;font-size:0.82rem">1,662&nbsp;/&nbsp;997 tokens</span>
        </div>
      </td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:top">Retrieved</td>
      <td style="vertical-align:top"><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">main.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">tests/test_formatter.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">tests/test_formatter_gen.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">tests/test_parser.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">utils/formatter.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">utils/parser.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">utils/validator.py</code></div></td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Strategy</td>
      <td style="color:#8250df;font-weight:bold">compact</td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Prompt&nbsp;size</td>
      <td style="color:#57606a">1,662 tokens</td>
    </tr>
  </table>
</div>



<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">Compaction log</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>


      Compaction triggered: 1,662 / 997 tokens (167%)
        compacted tests/test_parser.py: 279t → 173t  (saved 106t)
        compacted tests/test_formatter_gen.py: 255t → 165t  (saved 90t)
        compacted tests/test_formatter.py: 168t → 125t  (saved 43t)
        compacted main.py: 173t → 147t  (saved 26t)
        compacted utils/validator.py: 261t → 160t  (saved 101t)
        compacted utils/parser.py: 303t → 157t  (saved 146t)
      After compaction: 1,150 / 997 tokens (115%)



<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">After compaction</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>




<div style="border:1px solid #d0d7de;border-radius:8px;padding:14px 18px;
            margin:8px 0;background:#f6f8fa;font-family:monospace;font-size:0.88rem">
  <div style="font-weight:bold;color:#0969da;margin-bottom:10px">
    Pocket Agent&nbsp;&nbsp;·&nbsp;&nbsp;<em style="font-weight:normal;color:#57606a">How does the formatter handle missing fields?</em>
  </div>
  <table style="border-collapse:collapse;width:100%">
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:middle;white-space:nowrap">Token Budget</td>
      <td style="vertical-align:middle">
        <div style="display:inline-flex;align-items:center;gap:10px">
          <div style="width:200px;height:10px;border-radius:5px;background:#e0e0e0;overflow:hidden">
            <div style="width:200px;height:100%;background:#cf222e;transition:width 0.3s"></div>
          </div>
          <span style="color:#57606a;font-size:0.82rem">1,150&nbsp;/&nbsp;997 tokens</span>
        </div>
      </td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:top">Retrieved</td>
      <td style="vertical-align:top"><div style="margin:2px 0"><span style="background:#0969da;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">COLD</span>&nbsp;&nbsp;<code style="font-size:0.85rem">main.py</code></div><div style="margin:2px 0"><span style="background:#0969da;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">COLD</span>&nbsp;&nbsp;<code style="font-size:0.85rem">tests/test_formatter.py</code></div><div style="margin:2px 0"><span style="background:#0969da;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">COLD</span>&nbsp;&nbsp;<code style="font-size:0.85rem">tests/test_formatter_gen.py</code></div><div style="margin:2px 0"><span style="background:#0969da;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">COLD</span>&nbsp;&nbsp;<code style="font-size:0.85rem">tests/test_parser.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">utils/formatter.py</code></div><div style="margin:2px 0"><span style="background:#0969da;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">COLD</span>&nbsp;&nbsp;<code style="font-size:0.85rem">utils/parser.py</code></div><div style="margin:2px 0"><span style="background:#0969da;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">COLD</span>&nbsp;&nbsp;<code style="font-size:0.85rem">utils/validator.py</code></div></td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Strategy</td>
      <td style="color:#8250df;font-weight:bold">compact</td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Prompt&nbsp;size</td>
      <td style="color:#57606a">1,150 tokens</td>
    </tr>
  </table>
</div>


---
## Chapter 9 — Semantic RAG

**Goal:** retrieve files by meaning, not just keywords or filenames.

Grep and fuzzy scoring both rely on surface-level text overlap.
They miss cases like: *"where is the nesting logic handled?"* — the relevant
function is `_flatten` in `parser.py`, but neither the query nor the filename
contains the word "flatten".

Semantic retrieval fixes this by:
1. **Embedding** each file (or chunk) into a vector using `nomic-embed-text`
2. **Embedding** the query into the same vector space at runtime
3. **Ranking** files by cosine similarity — close vectors = similar meaning

The embeddings are computed once and cached in memory. On subsequent calls
only the (cheap) query embedding is recomputed.

**You will:**
- Write `embed()` — call Ollama's embedding endpoint
- Write `build_index()` — embed all files and store vectors in memory
- Write `semantic_retrieve()` — embed the query and return ranked files
- Compare semantic vs grep for a query that grep misses

### 9.1 Dependencies

Chapter 9 needs `numpy` for the cosine similarity calculation.
Make sure `nomic-embed-text` is pulled — this is the `OLLAMA_EMBED` model
set in the config cell at the top.


```python
import subprocess, sys

_CH8_DEPS = ["numpy"]
for _pkg in _CH8_DEPS:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", _pkg],
        stdout=subprocess.DEVNULL,
    )

import numpy as np
print("Chapter 9 dependencies ready:", _CH8_DEPS)

# Confirm the embed model is available
ok, models = ping_ollama()
if ok and any(OLLAMA_EMBED in m for m in models):
    print(f"Embed model '{OLLAMA_EMBED}' is available.")
else:
    print(f"WARNING: embed model '{OLLAMA_EMBED}' not found.")
    print(f"Run: ollama pull {OLLAMA_EMBED}")
```

    Chapter 9 dependencies ready: ['numpy']
    Embed model 'nomic-embed-text' is available.


### 9.2 `embed()` — call Ollama's embedding endpoint

Ollama exposes `/api/embed` (or `/api/embeddings` in older versions).
It accepts a model name and a string, returns a float vector.
We normalise the vector to unit length immediately so cosine similarity
later is just a dot product.


```python
def embed(text: str, model: str = OLLAMA_EMBED) -> np.ndarray:
    """
    Return a unit-normalised embedding vector for *text*.

    Uses Ollama's /api/embed endpoint (introduced in Ollama 1.1.26).
    Falls back to /api/embeddings for older installations.
    """
    payload = {"model": model, "input": text}
    try:
        r = requests.post(f"{OLLAMA_BASE_URL}/api/embed",
                          json=payload, timeout=60)
        r.raise_for_status()
        vec = r.json()["embeddings"][0]
    except Exception:
        # Fallback for older Ollama versions
        payload_old = {"model": model, "prompt": text}
        r = requests.post(f"{OLLAMA_BASE_URL}/api/embeddings",
                          json=payload_old, timeout=60)
        r.raise_for_status()
        vec = r.json()["embedding"]

    arr  = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 0 else arr


# Quick smoke test
_v = embed("hello world")
print(f"Vector dimension : {len(_v)}")
print(f"VVector norm     :  {np.linalg.norm(_v):.6f}  (unit-normalized — good for cosine similarity)")
```

    Vector dimension : 768
    VVector norm     :  1.000000  (unit-normalized — good for cosine similarity)


### 9.3 `build_index()` — embed every file, cache in memory

We chunk each file's content, embed every chunk, and store all the vectors
alongside the file metadata.  At query time we score a file by its *best*
matching chunk — so a relevant function buried at the end of a long file
still gets found.

For small files (like our sample project) one chunk covers the whole file,
so the behaviour is identical to a single embed call.  Cost scales linearly
with file size, not repo size.

> **Cost warning** — `build_index` makes one embedding call *per chunk per
> file*, and the result lives only in memory.  For a large repo with thousands
> of files this becomes slow and memory-hungry on every process restart.
>
> Production systems solve this with:
> - **Persistent vector stores** — ChromaDB, FAISS, pgvector, Weaviate —
>   embed once, persist to disk, query forever.
> - **Approximate nearest-neighbour (ANN) search** — sub-linear query time
>   instead of scanning every vector linearly.
> - **Orchestration libraries** — LlamaIndex, LangChain — handle chunking,
>   indexing, and retrieval behind a clean API.
>
> In Chapter 16 we rebuild this exact pipeline with LlamaIndex so you can
> see the 1-to-1 mapping between what we wrote here and what the library
> provides — and understand *why* the library exists rather than just
> cargo-culting it.


```python
# In-memory index: repo_root → list of {"path", "vectors", "bytes", "hot"}
_EMBED_INDEX: dict[str, list[dict]] = {}

CHUNK_SIZE    = 4000   # characters per chunk
CHUNK_OVERLAP = 200    # overlap between consecutive chunks


def embed_chunks(text: str) -> list[np.ndarray]:
    """
    Split *text* into overlapping chunks and embed each one.
    Returns a list of unit-norm vectors (one per chunk).
    Single-chunk files produce a one-element list — identical cost to before.
    """
    vecs  = []
    start = 0
    while start < len(text):
        chunk = text[start:start + CHUNK_SIZE]
        vec   = embed(chunk)
        if len(vec) > 0:
            vecs.append(vec)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return vecs


def build_index(
    repo_root:  str  = REPO_ROOT,
    extensions: set  = CODE_EXTENSIONS,
    force:      bool = REBUILD_INDEX,
) -> list[dict]:
    """
    Embed every source file under *repo_root* and cache the result.

    Each index entry stores *all* chunk vectors for the file.
    semantic_retrieve() scores by the best-matching chunk.

    Returns the index (list of dicts with a "vectors" key).
    On subsequent calls the cached index is returned unless *force=True*.
    """
    if repo_root in _EMBED_INDEX and not force:
        return _EMBED_INDEX[repo_root]

    files = glob_files("*", repo_root=repo_root)
    files = [f for f in files
             if Path(f["path"]).suffix.lower() in extensions]

    index = []
    for f in files:
        full_path = Path(repo_root) / f["path"]
        try:
            content = full_path.read_text(errors="ignore")
        except OSError:
            continue

        if not content.strip():
            continue

        vecs = embed_chunks(content)
        if not vecs:
            continue

        n_chunks = len(vecs)
        index.append({**f, "vectors": vecs, "content": content,
                      "tokens": count_tokens(content)})
        print(f"  embedded  {f['path']}  ({n_chunks} chunk{'s' if n_chunks > 1 else ''})")

    _EMBED_INDEX[repo_root] = index
    return index


show_rule("Building semantic index for sample_project")
idx = build_index(repo_root="sample_project", force=True)
print(f"\nIndex contains {len(idx)} files.")
```


<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">Building semantic index for sample_project</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>


      embedded  main.py  (1 chunk)
      embedded  tests/test_formatter.py  (1 chunk)
      embedded  tests/test_formatter_gen.py  (1 chunk)
      embedded  tests/test_parser.py  (1 chunk)
      embedded  utils/formatter.py  (1 chunk)
      embedded  utils/parser.py  (1 chunk)
      embedded  utils/validator.py  (1 chunk)
    
    Index contains 7 files.


### 9.4 `semantic_retrieve()` — rank by cosine similarity

Cosine similarity between unit vectors is just a dot product.
We embed the query, dot it against every file vector, and sort descending.
The result dict shape matches what `budget_load()` and `show_panel()` expect.


```python
def semantic_retrieve(
    query:     str,
    repo_root: str = REPO_ROOT,
    top_k:     int = 5,
) -> list[dict]:
    """
    Return the top-*k* files from the semantic index, ranked by cosine
    similarity to *query*.

    Each returned dict has a "score" key (cosine similarity, 0–1).
    """
    index     = build_index(repo_root=repo_root)
    query_vec = embed(query)

    if len(query_vec) == 0:
        # Query embedding failed (e.g. empty query) — fall back to empty list
        return []

    scored = []
    for entry in index:
        # Score by best-matching chunk — a relevant section anywhere in the
        # file wins, not just the beginning.
        chunk_sims = [
            float(np.dot(query_vec, vec))
            for vec in entry.get("vectors", [entry.get("vector", np.array([]))])
            if len(vec) > 0 and vec.shape == query_vec.shape
        ]
        if not chunk_sims:
            continue
        sim = max(chunk_sims)
        scored.append({
            "path":    entry["path"],
            "bytes":   entry["bytes"],
            "tokens":  entry["tokens"],
            "content": entry["content"],
            "hot":     False,
            "score":   round(sim, 4),
        })

    return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]


# ── Demo: the query grep struggles with ──────────────────────────────────────
query = "Where is the nesting logic handled?"

sem_results  = semantic_retrieve(query, repo_root="sample_project")
grep_results = grep_rank(query, repo_root="sample_project")

rows = ""
for i, f in enumerate(sem_results, 1):
    rows += f'<tr><td>semantic</td><td style="text-align:right">{i}</td><td style="text-align:right;color:#2da44e">{f["score"]}</td><td><code>{f["path"]}</code></td></tr>'
for i, f in enumerate(grep_results, 1):
    rows += f'<tr><td>grep</td><td style="text-align:right">{i}</td><td style="text-align:right;color:#bf8700">{f["score"]}</td><td><code>{f["path"]}</code></td></tr>'

display(HTML(f"""
<table style="border-collapse:collapse;font-family:monospace;font-size:0.9rem;width:100%">
  <caption style="font-weight:bold;margin-bottom:6px">Semantic vs Grep</caption>
  <thead><tr style="border-bottom:2px solid #d0d7de">
    <th style="text-align:left;padding:4px 12px">Method</th>
    <th style="text-align:right;padding:4px 12px">Rank</th>
    <th style="text-align:right;padding:4px 12px">Score</th>
    <th style="text-align:left;padding:4px 12px">File</th>
  </tr></thead>
  <tbody>{rows}</tbody>
</table>
"""))
```



<table style="border-collapse:collapse;font-family:monospace;font-size:0.9rem;width:100%">
  <caption style="font-weight:bold;margin-bottom:6px">Semantic vs Grep</caption>
  <thead><tr style="border-bottom:2px solid #d0d7de">
    <th style="text-align:left;padding:4px 12px">Method</th>
    <th style="text-align:right;padding:4px 12px">Rank</th>
    <th style="text-align:right;padding:4px 12px">Score</th>
    <th style="text-align:left;padding:4px 12px">File</th>
  </tr></thead>
  <tbody><tr><td>semantic</td><td style="text-align:right">1</td><td style="text-align:right;color:#2da44e">0.5435</td><td><code>main.py</code></td></tr><tr><td>semantic</td><td style="text-align:right">2</td><td style="text-align:right;color:#2da44e">0.5421</td><td><code>utils/validator.py</code></td></tr><tr><td>semantic</td><td style="text-align:right">3</td><td style="text-align:right;color:#2da44e">0.5225</td><td><code>tests/test_parser.py</code></td></tr><tr><td>semantic</td><td style="text-align:right">4</td><td style="text-align:right;color:#2da44e">0.5159</td><td><code>utils/formatter.py</code></td></tr><tr><td>semantic</td><td style="text-align:right">5</td><td style="text-align:right;color:#2da44e">0.5126</td><td><code>tests/test_formatter_gen.py</code></td></tr></tbody>
</table>



### 9.5 Try it end-to-end

Ask the nesting query. Semantic retrieval finds `parser.py` at the top
because `_flatten` is semantically close to "nesting logic",
even though neither word appears in the other.


```python
query = "Where is the nesting logic handled?"
repo  = "sample_project"

candidates   = semantic_retrieve(query, repo_root=repo)
manifest     = load_manifest(repo)
manifest_tok = manifest["tokens"]
loaded_files = budget_load(candidates, already_used=manifest_tok, repo_root=repo)

hot_files   = [f for f in loaded_files if f["hot"]]
prompt_size = manifest_tok + sum(f["tokens"] for f in hot_files) + count_tokens(query)

show_rule("Chapter 9 — semantic retrieval before LLM call")

reply, tokens_used = ask_with_manifest(query, repo_root=repo, files=hot_files)

show_rule("After call")
show_panel(
    query       = query,
    token_used  = tokens_used,
    files       = [{"path": f["path"], "hot": f["hot"]} for f in loaded_files],
    strategy    = "semantic RAG",
    prompt_size = prompt_size,
)
show_reply(reply)

```


<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">Chapter 9 — semantic retrieval before LLM call</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>




<div style="border:1px solid #d0d7de;border-radius:8px;padding:14px 18px;
            margin:8px 0;background:#f6f8fa;font-family:monospace;font-size:0.88rem">
  <div style="font-weight:bold;color:#0969da;margin-bottom:10px">
    Pocket Agent&nbsp;&nbsp;·&nbsp;&nbsp;<em style="font-weight:normal;color:#57606a">Where is the nesting logic handled?</em>
  </div>
  <table style="border-collapse:collapse;width:100%">
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:middle;white-space:nowrap">Token Budget</td>
      <td style="vertical-align:middle">
        <div style="display:inline-flex;align-items:center;gap:10px">
          <div style="width:200px;height:10px;border-radius:5px;background:#e0e0e0;overflow:hidden">
            <div style="width:0px;height:100%;background:#2da44e;transition:width 0.3s"></div>
          </div>
          <span style="color:#57606a;font-size:0.82rem">1,252&nbsp;/&nbsp;262,144 tokens</span>
        </div>
      </td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:top">Retrieved</td>
      <td style="vertical-align:top"><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">main.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">utils/validator.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">tests/test_parser.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">utils/formatter.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">tests/test_formatter_gen.py</code></div></td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Strategy</td>
      <td style="color:#8250df;font-weight:bold">manifest</td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Prompt&nbsp;size</td>
      <td style="color:#57606a">1,252 tokens</td>
    </tr>
  </table>
</div>



<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">After call</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>




<div style="border:1px solid #d0d7de;border-radius:8px;padding:14px 18px;
            margin:8px 0;background:#f6f8fa;font-family:monospace;font-size:0.88rem">
  <div style="font-weight:bold;color:#0969da;margin-bottom:10px">
    Pocket Agent&nbsp;&nbsp;·&nbsp;&nbsp;<em style="font-weight:normal;color:#57606a">Where is the nesting logic handled?</em>
  </div>
  <table style="border-collapse:collapse;width:100%">
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:middle;white-space:nowrap">Token Budget</td>
      <td style="vertical-align:middle">
        <div style="display:inline-flex;align-items:center;gap:10px">
          <div style="width:200px;height:10px;border-radius:5px;background:#e0e0e0;overflow:hidden">
            <div style="width:1px;height:100%;background:#2da44e;transition:width 0.3s"></div>
          </div>
          <span style="color:#57606a;font-size:0.82rem">1,661&nbsp;/&nbsp;262,144 tokens</span>
        </div>
      </td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:top">Retrieved</td>
      <td style="vertical-align:top"><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">main.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">utils/validator.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">tests/test_parser.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">utils/formatter.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">tests/test_formatter_gen.py</code></div></td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Strategy</td>
      <td style="color:#8250df;font-weight:bold">semantic RAG</td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Prompt&nbsp;size</td>
      <td style="color:#57606a">1,199 tokens</td>
    </tr>
  </table>
</div>



The **nesting logic**—specifically, the flattening of nested JSON structures—is handled in the **`utils/parser.py`** file, particularly in the `_flatten` function.

### Why?
In your `tests/test_parser.py`, there's a test case:
```python
def test_nested_dict(self):
    assert _flatten({"a": {"b": 1}}) == {"a.b": 1}
```
and:
```python
def test_deeply_nested(self):
    assert _flatten({"a": {"b": {"c": 42}}}) == {"a.b.c": 42}
```

These show that `_flatten` recursively handles nested dictionaries by converting them into dot-notation flat keys (e.g., `{"a": {"b": 1}}` → `{"a.b": 1}`).

While `_flatten` is not shown in your provided files, it must be defined in `utils/parser.py`, alongside `parse_json`, which likely:
1. Parses JSON text into Python objects.
2. Wraps single objects or non-dict lists into record lists.
3. Applies `_flatten` to ensure all records are flat dictionaries (as required by `to_csv`).

### Supporting evidence:
- `utils/formatter.py` assumes records are **flat dicts** (e.g., "list of flat dicts" in its docstring).
- `main.py` passes parsed records directly to `to_csv`.
- Test coverage in `test_parser.py` explicitly tests flattening behavior.
- No flattening logic appears in `formatter.py`, `validator.py`, or `main.py`.

✅ **Conclusion:** Nesting logic (flattening) is in `utils/parser.py`, specifically in the `_flatten` helper function used by `parse_json`.


---
## Chapter 10 — Full Pipeline

**Goal:** one function call, `run(query, repo)`, does everything.

Every chapter so far introduced one tool. Chapter 10 wires them together:

```
query
  │
  ├─ manifest loaded (always)
  │
  ├─ strategy picker
  │     ├─ "semantic"  → semantic_retrieve()
  │     ├─ "grep"      → grep_rank()
  │     └─ "fuzzy"     → rank_files(glob_files())
  │
  ├─ budget_load()   — JIT read, HOT/COLD split
  │
  ├─ compact()       — if > COMPACT_THRESHOLD
  │
  └─ ask_with_manifest()  → reply
```

The **strategy picker** is a small classifier that looks at the query:
- Contains a specific symbol or quoted string → grep (looking for code)
- Vague / conceptual phrasing → semantic (understanding meaning)
- Everything else → fuzzy (fast, good enough for named-file queries)

**You will:**
- Write `pick_strategy()` — classify the query
- Write `run()` — the full pipeline in one call
- Try three queries that each route to a different strategy

### 10.1 `pick_strategy()` — classify the query

A lightweight heuristic — no ML needed.
The three signals are mutually exclusive and checked in priority order.


```python
# Patterns that suggest the user is looking for a specific symbol or string
_GREP_SIGNALS = re.compile(
    r'(?:'
    r'"[^"]+"'              # quoted string
    r'|`[^`]+`'             # backtick symbol
    r'|raise\s+\w+'         # raise SomeError
    r'|def\s+\w+'           # def function_name
    r'|class\s+\w+'         # class ClassName
    r'|import\s+\w+'        # import something
    r'|\b[A-Z][a-zA-Z]+Error\b'  # ExceptionName
    r')'
)

# Words that suggest the user is asking about a concept, not looking for a symbol
_CONCEPTUAL_WORDS = {
    "why", "how", "explain", "what", "design", "architecture",
    "pattern", "approach", "concept", "idea", "strategy", "logic",
    "purpose", "reason", "difference", "relationship",
}

def pick_strategy(query: str) -> str:
    """
    Classify *query* into one of three retrieval strategies:
      "grep"     — looking for a specific symbol / string in code
      "semantic" — conceptual / meaning-based question
      "fuzzy"    — everything else (filename-level search)
    """
    # Grep signal: backticks, quotes, error names, def/class/import
    if _GREP_SIGNALS.search(query):
        return "grep"

    # Semantic signal: conceptual vocabulary
    words = set(query.lower().split())
    if words & _CONCEPTUAL_WORDS:
        return "semantic"

    return "fuzzy"


# ── Check the three example queries ──────────────────────────────────────────
for q in [
    "Where does the code raise a `TypeError`?",
    "Why does the parser wrap scalars in a dict?",
    "How does the formatter handle missing fields?",
]:
    print(f"  {pick_strategy(q):8s}  {q}")
```

      grep      Where does the code raise a `TypeError`?
      semantic  Why does the parser wrap scalars in a dict?
      semantic  How does the formatter handle missing fields?


### 10.2 `run()` — the full pipeline

This is the function all later chapters call.
It returns a `RunResult` named tuple so callers can inspect
the files that were loaded, the strategy used, and the token counts.


```python
from typing import NamedTuple


class RunResult(NamedTuple):
    reply:       str
    strategy:    str
    files:       list[dict]   # full list, HOT and COLD
    tokens_used: int
    compact_log: list[str]    # empty if compaction didn't fire


def run(
    query:     str,
    repo_root: str  = REPO_ROOT,
    strategy:  str  = "auto",   # "auto" | "fuzzy" | "grep" | "semantic"
    top_k:     int  = 8,
) -> RunResult:
    """
    Full retrieval + generation pipeline.

    Parameters
    ----------
    query     : the user's question
    repo_root : path to the repository to search
    strategy  : retrieval strategy; "auto" lets pick_strategy() decide
    top_k     : max candidates to consider before budget_load()
    """
    # ── 1. Manifest ─────────────────────────────────────────────────────────
    manifest     = load_manifest(repo_root)
    manifest_tok = manifest["tokens"]

    # ── 2. Strategy selection ────────────────────────────────────────────────
    strat = pick_strategy(query) if strategy == "auto" else strategy

    # ── 3. Retrieval ─────────────────────────────────────────────────────────
    if strat == "grep":
        candidates = grep_rank(query, repo_root=repo_root)
        # Append fuzzy extras for files with zero grep hits
        found_paths = {f["path"] for f in candidates}
        extras = [f for f in rank_files(glob_files("*.py", repo_root=repo_root), query)
                  if f["path"] not in found_paths]
        candidates = (candidates + extras)[:top_k]

    elif strat == "semantic":
        candidates = semantic_retrieve(query, repo_root=repo_root, top_k=top_k)

    else:   # fuzzy
        candidates = rank_files(glob_files("*.py", repo_root=repo_root), query)[:top_k]

    # ── 4. Budget-aware JIT load ─────────────────────────────────────────────
    loaded = budget_load(candidates, already_used=manifest_tok, repo_root=repo_root)

    # ── 5. Compaction (if needed) ─────────────────────────────────────────────
    hot_tok = sum(f.get("tokens", 0) for f in loaded if f["hot"])
    total   = manifest_tok + hot_tok + count_tokens(query)
    loaded, total, compact_log = compact(loaded, query, total)

    # ── 6. Generate reply ─────────────────────────────────────────────────────
    hot_files = [f for f in loaded if f["hot"]]
    reply, tokens_used = ask_with_manifest(query, repo_root=repo_root, files=hot_files)

    return RunResult(
        reply        = reply,
        strategy     = strat,
        files        = loaded,
        tokens_used  = tokens_used,
        compact_log  = compact_log,
    )
```

### 10.3 Try all three strategies

Three queries, each routed automatically to a different strategy.
Watch the panel's **Strategy** field change on each run.


```python
repo = "sample_project"

queries = [
    "Where does the code raise a `TypeError`?",           # → grep
    "Why does the parser wrap scalars in a dict?",        # → semantic
    "What does the formatter do with the delimiter?",     # → fuzzy
]

for q in queries:
    show_rule(f"query: {q}")
    result = run(q, repo_root=repo)
    print(f"\nStrategy used: {result.strategy}")
    show_reply(result.reply)

```


<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">query: Where does the code raise a `TypeError`?</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>




<div style="border:1px solid #d0d7de;border-radius:8px;padding:14px 18px;
            margin:8px 0;background:#f6f8fa;font-family:monospace;font-size:0.88rem">
  <div style="font-weight:bold;color:#0969da;margin-bottom:10px">
    Pocket Agent&nbsp;&nbsp;·&nbsp;&nbsp;<em style="font-weight:normal;color:#57606a">Where does the code raise a `TypeError`?</em>
  </div>
  <table style="border-collapse:collapse;width:100%">
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:middle;white-space:nowrap">Token Budget</td>
      <td style="vertical-align:middle">
        <div style="display:inline-flex;align-items:center;gap:10px">
          <div style="width:200px;height:10px;border-radius:5px;background:#e0e0e0;overflow:hidden">
            <div style="width:1px;height:100%;background:#2da44e;transition:width 0.3s"></div>
          </div>
          <span style="color:#57606a;font-size:0.82rem">1,743&nbsp;/&nbsp;262,144 tokens</span>
        </div>
      </td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:top">Retrieved</td>
      <td style="vertical-align:top"><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">utils/parser.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">utils/validator.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">tests/test_parser.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">utils/formatter.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">main.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">tests/test_formatter.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">tests/test_formatter_gen.py</code></div></td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Strategy</td>
      <td style="color:#8250df;font-weight:bold">manifest</td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Prompt&nbsp;size</td>
      <td style="color:#57606a">1,743 tokens</td>
    </tr>
  </table>
</div>


    
    Strategy used: grep



The code raises a `TypeError` in **`utils/validator.py`**, specifically inside the `validate_schema` function.

Here's exactly where and why:

### `TypeError` location:
In `utils/validator.py`, inside the `validate_schema` function:
```python
if expected and not isinstance(record[field], expected):
    raise TypeError(
        f"Record {i}: field '{field}' expected {expected_type_name}, "
        f"got {type(record[field]).__name__}"
    )
```

### Why it raises `TypeError`:
The `validate_schema` function explicitly raises a `TypeError` when:
1. A required field is missing → `raise TypeError(... "missing required field ...")`
2. A field’s value is not of the expected type (e.g., expecting `str` but getting `int`, or expecting `bool` but getting `str`) → `raise TypeError(... "expected {type}, got {actual_type}")`

These are **intentional type validation checks**, and `TypeError` is used because it semantically matches *type mismatches* — Python’s built-in convention (e.g., `isinstance(x, str)` failing may raise `TypeError` if `str` isn't a valid type, but here it's used explicitly).

### Important: Where *not* to look for `TypeError`:
- `utils/parser.py`: Raises `ValueError` only (never `TypeError`)
- `utils/formatter.py`: Does not raise `TypeError` — it assumes valid input (but could raise `TypeError` if given non-dict in `records`, but this is not handled or tested)
- `main.py`: Depends on above; could propagate `TypeError` from `validate_schema`, but doesn’t define its own raise
- Tests: none of the tests expect or cause `TypeError` (they use `pytest.raises(ValueError)` for parser errors, and none expect `TypeError`)

### But wait — could `isinstance(record[field], expected)` raise `TypeError`?
Yes, in theory:
```python
isinstance(value, expected)
```
can raise `TypeError` **only if** `expected` is **not a type or tuple of types**, e.g.:
```python
isinstance(x, "string")  # ❌ TypeError: isinstance() arg 2 must be a type or tuple of types
```

But in this code:
```python
type_map = {"string": str, "number": (int, float), "boolean": bool}
expected = type_map.get(expected_type_name)  # → always str, (int,float), or bool
```
So `expected` is always a valid type/tuple-of-types. Thus, no *unexpected* `TypeError` from `isinstance`.

However, if `expected_type_name` is invalid (e.g., `"foo"` not in `type_map`), then:
```python
expected = type_map.get("foo")  # → None
if expected and ...  # short-circuits, no isinstance called → safe
```

So **only intentional `TypeError`s are raised** in `validate_schema`.

✅ **Answer**:  
A `TypeError` is raised **in `utils/validator.py` inside `validate_schema()`**, when:
- A required field is missing **or**
- A field’s actual type doesn’t match the expected type (e.g., `"string"` expected but got `int`).



<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">query: Why does the parser wrap scalars in a dict?</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>




<div style="border:1px solid #d0d7de;border-radius:8px;padding:14px 18px;
            margin:8px 0;background:#f6f8fa;font-family:monospace;font-size:0.88rem">
  <div style="font-weight:bold;color:#0969da;margin-bottom:10px">
    Pocket Agent&nbsp;&nbsp;·&nbsp;&nbsp;<em style="font-weight:normal;color:#57606a">Why does the parser wrap scalars in a dict?</em>
  </div>
  <table style="border-collapse:collapse;width:100%">
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:middle;white-space:nowrap">Token Budget</td>
      <td style="vertical-align:middle">
        <div style="display:inline-flex;align-items:center;gap:10px">
          <div style="width:200px;height:10px;border-radius:5px;background:#e0e0e0;overflow:hidden">
            <div style="width:1px;height:100%;background:#2da44e;transition:width 0.3s"></div>
          </div>
          <span style="color:#57606a;font-size:0.82rem">1,744&nbsp;/&nbsp;262,144 tokens</span>
        </div>
      </td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:top">Retrieved</td>
      <td style="vertical-align:top"><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">utils/parser.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">tests/test_parser.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">utils/formatter.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">main.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">tests/test_formatter_gen.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">tests/test_formatter.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">utils/validator.py</code></div></td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Strategy</td>
      <td style="color:#8250df;font-weight:bold">manifest</td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Prompt&nbsp;size</td>
      <td style="color:#57606a">1,744 tokens</td>
    </tr>
  </table>
</div>


    
    Strategy used: semantic



The parser wraps scalars in a dict (e.g., `1` → `{"value": 1}`) to **ensure uniform output format** — specifically, to guarantee that `parse_json` **always returns a list of flat dictionaries**, regardless of whether the input JSON is:

- a list of objects → returned as-is (already dicts)  
- a single object → wrapped in a list (so result is `[dict]`)  
- **a list of scalars** → each scalar needs to become a dict to match the expected record format.

### Why is this necessary?

The **entire pipeline assumes structured data**:

1. `parse_json(raw)` → must return `list[dict]`
2. `validate_schema(records, schema)` → expects each `record` to be a dict with fields to validate
3. `to_csv(records)` → expects each record to be a dict with consistent/determinable headers

If you passed `[1, 2, 3]`, and didn’t wrap scalars, you’d get `list[int]`, which would:
- Break `validate_schema` (no fields to check!),
- Break `to_csv` (no keys → no headers, or chaos when writing non-dict rows).

### Why `"value"` specifically?
- It’s a **sensible generic key** for scalar data.
- Avoids name collisions in real data (unlikely someone has a field literally named `"value"` unless intentional).
- Makes it explicit: `"this was originally a bare value"`.

### Alternative designs considered (and why they’re worse):
| Approach | Problem |
|--------|---------|
| Return mixed types: `list[dict | scalar]` | Breaks downstream functions (`to_csv`, `validate_schema`). |
| Skip wrapping and raise error on scalar list | Less flexible — users might legitimately want to convert `[1,2,3]` to CSV (1 column). |
| Use a different key like `__value__` | Unnecessarily obscure; `"value"` is clear and minimal. |

### Example:
```python
# Input: '[1, 2, 3]'
# Output: [{'value': 1}, {'value': 2}, {'value': 3}]
# → CSV: "value\r\n1\r\n2\r\n3\r\n"
```
This is a clean, predictable way to handle scalar lists while preserving consistency.

✅ So: **wrapping scalars maintains the `list[dict]` contract**, enabling robust downstream processing.



<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">query: What does the formatter do with the delimiter?</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>




<div style="border:1px solid #d0d7de;border-radius:8px;padding:14px 18px;
            margin:8px 0;background:#f6f8fa;font-family:monospace;font-size:0.88rem">
  <div style="font-weight:bold;color:#0969da;margin-bottom:10px">
    Pocket Agent&nbsp;&nbsp;·&nbsp;&nbsp;<em style="font-weight:normal;color:#57606a">What does the formatter do with the delimiter?</em>
  </div>
  <table style="border-collapse:collapse;width:100%">
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:middle;white-space:nowrap">Token Budget</td>
      <td style="vertical-align:middle">
        <div style="display:inline-flex;align-items:center;gap:10px">
          <div style="width:200px;height:10px;border-radius:5px;background:#e0e0e0;overflow:hidden">
            <div style="width:1px;height:100%;background:#2da44e;transition:width 0.3s"></div>
          </div>
          <span style="color:#57606a;font-size:0.82rem">1,744&nbsp;/&nbsp;262,144 tokens</span>
        </div>
      </td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:top">Retrieved</td>
      <td style="vertical-align:top"><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">tests/test_formatter_gen.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">tests/test_formatter.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">utils/formatter.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">main.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">utils/parser.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">tests/test_parser.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">utils/validator.py</code></div></td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Strategy</td>
      <td style="color:#8250df;font-weight:bold">manifest</td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Prompt&nbsp;size</td>
      <td style="color:#57606a">1,744 tokens</td>
    </tr>
  </table>
</div>


    
    Strategy used: semantic



The formatter's `to_csv()` function uses the `delimiter` parameter to control how fields are separated in the output CSV.

Looking at the `to_csv()` function in `utils/formatter.py`:

```python
writer = csv.DictWriter(buf, fieldnames=headers,
                        delimiter=delimiter, extrasaction="ignore")
```

The `delimiter` is passed directly to Python's built-in `csv.DictWriter` constructor. This means:

1. **Default behavior**: When no delimiter is specified (or `delimiter=","`), fields are separated by commas, e.g., `"Alice,30,NYC"`

2. **Custom delimiter**: When you pass a different delimiter (like `";"` as in `test_to_csv_custom_delimiter`), it replaces the comma separator, e.g., `"Alice;30;NYC"`

3. **How it works**: The `csv.DictWriter` internally uses this delimiter when writing both the header row and each data row, ensuring consistent separation throughout the output.

The test `test_to_csv_custom_delimiter()` confirms this:
```python
result = to_csv(records, delimiter=";")
assert result == "name;score\r\nAlice;95\r\nBob;88\r\n"
```

So the formatter simply delegates the delimiter handling to Python's standard CSV library, which properly handles the delimiter in all relevant places (headers, data rows, and any necessary quoting of fields that contain the delimiter character).


---
## Chapter 11 — Write + Diff

**Goal:** build the write primitive — a function you call directly to propose
and apply a file change.

We are not inside an agent loop yet.  You decide what to change and when;
the LLM proposes the new content; you confirm before anything touches disk.
Chapter 12 wires this into the agent loop so the model can make that decision
on its own.

Two principles guide this chapter:

1. **Show before you write.** The diff is always printed before any file is touched.
2. **Full file, not a patch.** The LLM receives the original content and returns
   the complete new file.  We derive the diff from those two strings — no fragile
   line-number arithmetic.

**You will:**
- Write `make_diff()` — unified diff between two strings, rendered as colour HTML
- Write `apply_patch()` — ask the LLM to rewrite a file given a plain-English instruction
- Write `write_file()` — wraps both: show diff, then write on confirmation
- Try it: call `write_file()` directly to add type hints to a function

### 11.1 `make_diff()` — coloured unified diff

Python's `difflib.unified_diff` does the heavy lifting.
`print_diff()` renders the result as colour-coded HTML in the notebook:
green for additions, red for removals, blue for chunk headers.



```python
import difflib

def make_diff(
    original:  str,
    proposed:  str,
    file_path: str = "<file>",
    context:   int = 3,
) -> str:
    """
    Return a unified diff string between *original* and *proposed*.
    Returns an empty string if they are identical.
    """
    orig_lines = original.splitlines(keepends=True)
    new_lines  = proposed.splitlines(keepends=True)
    diff_lines = list(difflib.unified_diff(
        orig_lines, new_lines,
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        n=context,
    ))
    return "".join(diff_lines)


def print_diff(diff: str) -> None:
    """Render a unified diff with colour-coded lines in the notebook."""
    if not diff:
        print("No changes.")
        return
    lines = diff.splitlines()
    html_lines = []
    for ln in lines:
        safe = ln.replace("&","&amp;").replace("<","&lt;")
        if ln.startswith("+") and not ln.startswith("+++"):
            html_lines.append(f'<span style="color:#2da44e">{safe}</span>')
        elif ln.startswith("-") and not ln.startswith("---"):
            html_lines.append(f'<span style="color:#cf222e">{safe}</span>')
        elif ln.startswith("@@"):
            html_lines.append(f'<span style="color:#0969da">{safe}</span>')
        else:
            html_lines.append(f'<span style="color:#57606a">{safe}</span>')
    display(HTML(
        '<pre style="background:#f6f8fa;border:1px solid #d0d7de;border-radius:6px;'
        'padding:12px;font-size:0.82rem;overflow-x:auto;line-height:1.4">'
        + "\n".join(html_lines) + "</pre>"
    ))


# ── Quick demo ────────────────────────────────────────────────────────────────
_original = 'def add(a, b):\n    return a + b\n'
_proposed = 'def add(a: int, b: int) -> int:\n    """Return the sum of a and b."""\n    return a + b\n'
print_diff(make_diff(_original, _proposed, "math_utils.py"))
```


<pre style="background:#f6f8fa;border:1px solid #d0d7de;border-radius:6px;padding:12px;font-size:0.82rem;overflow-x:auto;line-height:1.4"><span style="color:#57606a">--- a/math_utils.py</span>
<span style="color:#57606a">+++ b/math_utils.py</span>
<span style="color:#0969da">@@ -1,2 +1,3 @@</span>
<span style="color:#cf222e">-def add(a, b):</span>
<span style="color:#2da44e">+def add(a: int, b: int) -> int:</span>
<span style="color:#2da44e">+    """Return the sum of a and b."""</span>
<span style="color:#57606a">     return a + b</span></pre>


### 11.2 `apply_patch()` — ask the LLM to rewrite a file

We give the model the original file content and a plain-English instruction.
It returns the **complete** new file — not a patch, not just the changed lines.
This is deliberate: asking the model to produce a full file is more reliable
than asking it to produce a diff, which it frequently gets wrong.


```python
def apply_patch(
    file_path:   str,
    instruction: str,
    repo_root:   str = REPO_ROOT,
) -> tuple[str, str]:
    """
    Ask the LLM to apply *instruction* to the file at *file_path*.

    Returns (original_content, proposed_content).
    The proposed content is the raw LLM output, stripped of markdown fences.
    If the file does not exist yet, original is treated as empty string
    (allows the agent to create new files).
    """
    # Normalise: strip repo_root prefix if the model included it
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
        f"FILE: {rel}\n"
        f"```\n{original}\n```"
    )

    proposed_raw, _ = chat([{"role": "user", "content": prompt}])

    # Strip markdown code fences if the model added them anyway
    proposed = re.sub(r"^```[a-zA-Z]*\n?", "", proposed_raw.strip())
    proposed = re.sub(r"\n?```$", "", proposed)

    return original, proposed.strip() + "\n"
```

### 11.3 `write_file()` — diff, confirm, write

`write_file()` wraps `apply_patch()`:
- Always shows the diff first
- In `dry_run=True` mode (default) it stops there — nothing is written
- In `dry_run=False` mode it writes the file after showing the diff


```python
def write_file(
    file_path:   str,
    instruction: str,
    repo_root:   str  = REPO_ROOT,
    dry_run:     bool = True,
) -> tuple[str, str, str]:
    """
    Apply *instruction* to *file_path*, show the diff, optionally write.

    Parameters
    ----------
    file_path   : path relative to repo_root
    instruction : plain-English change request
    repo_root   : repository root
    dry_run     : if True, print the diff but do NOT write to disk

    Returns (original, proposed, diff_text).
    """
    print(f"\nApplying: {instruction}")
    print(f"File: {file_path}\n")

    original, proposed = apply_patch(file_path, instruction, repo_root)
    diff = make_diff(original, proposed, file_path)

    show_rule("Proposed diff")
    print_diff(diff)

    if not diff:
        print("No changes proposed by the model.")
        return original, proposed, diff

    if dry_run:
        print("\nDRY RUN — file not written. Set dry_run=False to apply.")
    else:
        full_path = Path(repo_root) / file_path
        full_path.write_text(proposed)
        print(f"\n✓ Written: {full_path}")

    return original, proposed, diff
```

### 11.4 Try it — add type hints to `_flatten`

Call `write_file()` directly, just like you would from the REPL.
The LLM reads the file, proposes the change, and prints the diff.
Nothing is written until you set `dry_run=False`.

In Chapter 12 the agent loop will make this call automatically as part of a plan.


```python
original, proposed, diff = write_file(
    file_path   = "utils/parser.py",
    instruction = "Add PEP 484 type annotations to all function signatures. "
                  "Do not change any logic or add any comments.",
    repo_root   = "sample_project",
    dry_run     = True,    # ← change to False to write to disk
)
```

    
    Applying: Add PEP 484 type annotations to all function signatures. Do not change any logic or add any comments.
    File: utils/parser.py
    



<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">Proposed diff</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>



<pre style="background:#f6f8fa;border:1px solid #d0d7de;border-radius:6px;padding:12px;font-size:0.82rem;overflow-x:auto;line-height:1.4"><span style="color:#57606a">--- a/utils/parser.py</span>
<span style="color:#57606a">+++ b/utils/parser.py</span>
<span style="color:#0969da">@@ -1,9 +1,10 @@</span>
<span style="color:#57606a"> """Parse a JSON string into a list of flat dicts."""</span>
<span style="color:#57606a"> </span>
<span style="color:#57606a"> import json</span>
<span style="color:#2da44e">+from typing import Any</span>
<span style="color:#57606a"> </span>
<span style="color:#57606a"> </span>
<span style="color:#cf222e">-def parse_json(raw: str) -> list[dict]:</span>
<span style="color:#2da44e">+def parse_json(raw: str) -> list[dict[str, Any]]:</span>
<span style="color:#57606a">     """</span>
<span style="color:#57606a">     Accept a JSON string that is either:</span>
<span style="color:#57606a">     - a list of objects   → returned as-is</span>
<span style="color:#0969da">@@ -26,7 +27,7 @@</span>
<span style="color:#57606a">     raise ValueError(f"Expected a JSON object or array, got {type(data).__name__}")</span>
<span style="color:#57606a"> </span>
<span style="color:#57606a"> </span>
<span style="color:#cf222e">-def _flatten(obj: dict, prefix: str = "", sep: str = ".") -> dict:</span>
<span style="color:#2da44e">+def _flatten(obj: dict[str, Any], prefix: str = "", sep: str = ".") -> dict[str, Any]:</span>
<span style="color:#57606a">     """Recursively flatten nested dicts: {"a": {"b": 1}} → {"a.b": 1}."""</span>
<span style="color:#57606a">     result = {}</span>
<span style="color:#57606a">     for key, value in obj.items():</span></pre>


    
    DRY RUN — file not written. Set dry_run=False to apply.


---
## Chapter 12 — Agent Loop

**Goal:** turn the agent from a question-answerer into an autonomous task executor.

So far each chapter required the user to drive every step.
Chapter 12 adds a loop that:

1. **Plans** — asks the LLM to break the task into ordered steps
2. **Executes** — for each step, decides whether to read (→ `run()`) or write (→ `write_file()`)
3. **Verifies** — after writing, asks the LLM to confirm the change makes sense
4. **Iterates** — repeats until the plan is complete or `max_steps` is reached

The loop is intentionally simple — no tool-calling API, no JSON schema enforcement.
The LLM outputs plain text; the loop parses a lightweight convention:

```
STEP: <description>
ACTION: read | write
TARGET: <file path>   (for write actions)
INSTRUCTION: <what to change>   (for write actions)
```

**You will:**
- Write `plan_task()` — ask the LLM to emit a structured step list
- Write `execute_step()` — dispatch one step to read or write
- Write `agent_loop()` — run the full plan with a step cap and verification

### 12.1 `plan_task()` — break a task into steps

We give the model the manifest and a system prompt that enforces
the `STEP / ACTION / TARGET / INSTRUCTION` format.
The output is parsed into a list of step dicts.


```python
_PLAN_SYSTEM = """\
You are a coding agent. Given a task and a project map, produce an ordered
list of ALL steps needed to fully complete the task.

Each step must be one of these formats (blank line between steps):

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
ACTION: glob
PATTERN: <glob pattern, e.g. utils/**/*.py>

STEP: <one-sentence description>
ACTION: grep
PATTERN: <regex pattern to search file contents>

Rules:
- Prefer ACTION: bash for file system operations (copy, move, delete, rename).
- Use ACTION: glob to find files by name pattern.
- Use ACTION: grep to search file contents for a regex pattern.
- Use ACTION: read only when you need code understanding before you can write.
- Use ACTION: write when the LLM must generate or edit file content.
- Never invent file paths; never use absolute or /tmp paths.
- A single READ step retrieves from the whole codebase — ask about multiple files
  in one step rather than one step per file.
- Each WRITE step targets exactly one file.
- Be specific in INSTRUCTION — the change will be applied by another model with no extra context.
- Do not add any text outside the step blocks.
"""

```


```python
def plan_task(task: str, repo_root: str = REPO_ROOT) -> list[dict]:
    """
    Ask the LLM to produce a step-by-step plan for *task*.

    Injects the real file listing into the prompt so the model plans against
    actual filenames rather than invented placeholders.

    Returns a list of dicts:
        {"step": str, "action": "read"|"write"|"bash"|"glob"|"grep",
         "target": str, "instruction": str, "cmd": str, "pattern": str}
    """
    manifest  = load_manifest(repo_root)
    all_files = glob_files("*", repo_root=repo_root)
    file_list = "\n".join(f["path"] for f in all_files)

    prompt = (
        f"PROJECT MAP:\n{manifest['text']}\n\n"
        f"AVAILABLE FILES (use only if the task involves the codebase):\n{file_list}\n\n"
        f"TASK: {task}"
    )

    raw, _ = chat([
        {"role": "system", "content": _PLAN_SYSTEM},
        {"role": "user",   "content": prompt},
    ])

    steps = []
    # Split on every STEP: marker regardless of blank lines between steps
    for block in re.split(r"(?m)(?=^STEP:)", raw.strip()):
        if not block.strip():
            continue
        lines = {
            m.group(1).lower(): m.group(2).strip()
            for line in block.splitlines()
            if (m := re.match(
                r"^(STEP|ACTION|TARGET|INSTRUCTION|CMD|PATTERN):\s*(.+)$",
                line, re.IGNORECASE
            ))
        }
        action = lines.get("action", "").lower()
        has_content = (
            "target"  in lines or  # read / write
            "cmd"     in lines or  # bash
            "pattern" in lines or  # glob / grep
            action in ("read", "write", "bash", "glob", "grep")
        )
        if action and has_content:
            steps.append({
                "step":        lines.get("step", ""),
                "action":      action,
                "target":      lines.get("target", ""),
                "instruction": lines.get("instruction", ""),
                "cmd":         lines.get("cmd", ""),
                "pattern":     lines.get("pattern", ""),
            })

    return steps

```

### 12.2 Tools and `execute_step()`

Three tools give the agent the ability to interact with the environment directly,
rather than just reading and proposing changes:

| Tool | What it does |
|------|-------------|
| `bash` | Run any shell command — returns stdout, stderr, and exit code |
| `glob` | Find files by pattern — fast, no file content read |
| `grep` | Search file contents by regex — returns matching files with snippets |

`execute_step()` is the dispatcher: it routes each plan step to the right handler —
`run()` for reads, `write_file()` for writes, and `call_tool()` for tool steps.



```python
CURRENT_REPO_ROOT = REPO_ROOT
_WORKTREE_STACK   = []


def _resolve_root(repo_root: str | None = None) -> str:
    return repo_root or CURRENT_REPO_ROOT


def tool_bash(cmd: str, repo_root: str | None = None) -> dict:
    """Run a shell command in repo_root. Returns stdout, stderr, and exit code."""
    root   = _resolve_root(repo_root)
    result = subprocess.run(cmd, shell=True, cwd=root, capture_output=True, text=True)
    return {
        "cmd":    cmd,
        "cwd":    root,
        "code":   result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def tool_glob(pattern: str, repo_root: str | None = None) -> list[dict]:
    """Find files matching a glob pattern (e.g. '**/*.py'). Returns file metadata — no content."""
    return glob_files(pattern, repo_root=_resolve_root(repo_root))


def tool_grep(pattern: str, repo_root: str | None = None) -> list[dict]:
    """Search file contents for a regex pattern. Returns matching files with line snippets."""
    return grep_repo(pattern, repo_root=_resolve_root(repo_root))

```


```python
def execute_step(
    step:      dict,
    repo_root: str  = REPO_ROOT,
    dry_run:   bool = True,
) -> str:
    """
    Execute one plan step. Returns a short status string for the loop log.
    (Chapter 14 upgrades this to return a structured dict instead.)

    read  → run(target_as_query)
    write → write_file(target, instruction)
    bash  → tool_bash(cmd)
    glob  → tool_glob(pattern)
    grep  → tool_grep(pattern)
    """
    action = step.get("action", "read")

    if action == "write":
        write_file(
            file_path   = step["target"],
            instruction = step["instruction"],
            repo_root   = repo_root,
            dry_run     = dry_run,
        )
        return "wrote (dry)" if dry_run else "wrote"

    if action == "bash":
        result = tool_bash(step.get("cmd", ""), repo_root=repo_root)
        output = (result["stdout"] + result["stderr"]).strip()
        suffix = "..." if len(output) > 800 else ""
        print(f"\nBash result:\n{output[:800]}{suffix}\n")
        return f"bash: {'ok' if result['code'] == 0 else 'error'}"

    if action == "glob":
        result = tool_glob(step.get("pattern", "**/*"), repo_root=repo_root)
        print(f"\nGlob ({len(result)} files): " + ", ".join(f["path"] for f in result[:10]))
        return f"glob: {len(result)} files"

    if action == "grep":
        result = tool_grep(step.get("pattern", ""), repo_root=repo_root)
        print(f"\nGrep ({len(result)} matches): " + ", ".join(f["path"] for f in result[:10]))
        return f"grep: {len(result)} matches"

    # default: read
    result = run(step["target"], repo_root=repo_root)
    suffix = "..." if len(result.reply) > 400 else ""
    print(f"\nRead result:\n{result.reply[:400]}{suffix}\n")
    return "read"

```

### 12.3 `agent_loop()` — plan → execute → verify

The loop runs the full plan, printing progress at each step.
After all write steps complete, it runs a final verification query
to ask the model whether the changes look correct.


```python
def agent_loop(
    task:      str,
    repo_root: str  = REPO_ROOT,
    dry_run:   bool = True,
    max_steps: int  = 8,
) -> list[dict]:
    """
    Autonomous task execution loop.

    1. Plan the task with plan_task()
    2. Execute each step with execute_step()
    3. Verify written files exist on disk (skipped on dry_run)

    Returns the step log (list of dicts with "step", "action", "status").
    """
    show_rule(f"Agent Loop  ·  {task[:60]}")
    print(f"repo: {repo_root}  |  dry_run={dry_run}  |  max_steps={max_steps}\n")

    # ── 1. Plan ───────────────────────────────────────────────────────────────
    plan = plan_task(task, repo_root=repo_root)
    if not plan:
        print("Planning failed — no steps parsed from model output.")
        return []

    print(f"Plan: {len(plan)} step(s)\n")

    # ── 2. Execute ────────────────────────────────────────────────────────────
    log = []
    for i, step in enumerate(plan[:max_steps], 1):
        show_rule(f"Step {i}/{min(len(plan), max_steps)}  {step['action'].upper()}  {step['step']}")
        status = execute_step(step, repo_root=repo_root, dry_run=dry_run)
        log.append({**step, "status": status})
        print(f"↳ {status}")

    # ── 3. Verify ─────────────────────────────────────────────────────────────
    written = [s for s in log if s["action"] == "write"]
    if written and not dry_run:
        show_rule("Verification")
        all_ok = True
        for s in written:
            path = Path(repo_root) / s["target"]
            icon = "✅" if path.exists() else "❌"
            print(f"{icon} {s['target']}")
            if not path.exists():
                all_ok = False
        print("\nAll files confirmed on disk." if all_ok else "\nSome files are missing.")
    elif written and dry_run:
        print("\nDRY RUN — skipping verification (files were not written).")

    show_rule("Loop complete")
    return log

```

### 12.4 Try it

Run the agent loop on a real task.
`dry_run=True` means no files are touched — you can set it to `False`
once you've reviewed the diffs and are happy with the plan.

### 12.5 Tool demo — syntax check

`tool_bash` lets the agent act on the repo, not just read it.
Here we call it directly to check every Python file for syntax errors — the same
call the agent emits when its plan includes:

```
ACTION: tool
TOOL: bash
ARGS: {"cmd": "find . -name '*.py' -exec python -m py_compile {} +"}
```

`py_compile` is Python's built-in syntax checker. It exits with code 0 if every
file parses cleanly, or prints the offending line and exits with code 1.



```python
import sys as _sys

result = tool_bash(
    f"{_sys.executable} -m py_compile $(find . -name '*.py') 2>&1 && echo 'All OK'",
    repo_root="sample_project",
)

if result["code"] == 0:
    print("✓ All Python files pass syntax check.")
else:
    print("Syntax errors found:\n")
    print(result["stdout"] or result["stderr"])

```

    ✓ All Python files pass syntax check.



```python
log = agent_loop(
    task      = "Add type annotations to all functions in utils/ and update their docstrings.",
    repo_root = "sample_project",
    dry_run   = True,    # ← change to False to write files
    max_steps = 6,
)
```


<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">Agent Loop  ·  Add type annotations to all functions in utils/ and update t</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>


    repo: sample_project  |  dry_run=True  |  max_steps=6
    
    STEP: Read all utility files to understand current function signatures and docstrings
    ACTION: read
    TARGET: utils/formatter.py, utils/parser.py, utils/validator.py
    
    STEP: Add type annotations to all functions in utils/formatter.py and update their docstrings
    ACTION: write
    TARGET: utils/formatter.py
    INSTRUCTION: Add comprehensive type annotations to all function parameters and return values, and update docstrings to include parameter descriptions and return value descriptions following the existing docstring style.
    
    STEP: Add type annotations to all functions in utils/parser.py and update their docstrings
    ACTION: write
    TARGET: utils/parser.py
    INSTRUCTION: Add comprehensive type annotations to all function parameters and return values, and update docstrings to include parameter descriptions and return value descriptions following the existing docstring style.
    
    STEP: Add type annotations to all functions in utils/validator.py and update their docstrings
    ACTION: write
    TARGET: utils/validator.py
    INSTRUCTION: Add comprehensive type annotations to all function parameters and return values, and update docstrings to include parameter descriptions and return value descriptions following the existing docstring style.
    Plan: 4 step(s)
    



<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">Step 1/4  READ  Read all utility files to understand current function signatures and docstrings</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>




<div style="border:1px solid #d0d7de;border-radius:8px;padding:14px 18px;
            margin:8px 0;background:#f6f8fa;font-family:monospace;font-size:0.88rem">
  <div style="font-weight:bold;color:#0969da;margin-bottom:10px">
    Pocket Agent&nbsp;&nbsp;·&nbsp;&nbsp;<em style="font-weight:normal;color:#57606a">utils/formatter.py, utils/parser.py, utils/validator.py</em>
  </div>
  <table style="border-collapse:collapse;width:100%">
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:middle;white-space:nowrap">Token Budget</td>
      <td style="vertical-align:middle">
        <div style="display:inline-flex;align-items:center;gap:10px">
          <div style="width:200px;height:10px;border-radius:5px;background:#e0e0e0;overflow:hidden">
            <div style="width:1px;height:100%;background:#2da44e;transition:width 0.3s"></div>
          </div>
          <span style="color:#57606a;font-size:0.82rem">1,747&nbsp;/&nbsp;262,144 tokens</span>
        </div>
      </td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a;vertical-align:top">Retrieved</td>
      <td style="vertical-align:top"><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">utils/parser.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">utils/formatter.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">utils/validator.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">tests/test_parser.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">tests/test_formatter.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">main.py</code></div><div style="margin:2px 0"><span style="background:#cf222e;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.78rem;font-weight:bold">HOT</span>&nbsp;&nbsp;<code style="font-size:0.85rem">tests/test_formatter_gen.py</code></div></td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Strategy</td>
      <td style="color:#8250df;font-weight:bold">manifest</td>
    </tr>
    <tr>
      <td style="padding:4px 14px 4px 0;color:#57606a">Prompt&nbsp;size</td>
      <td style="color:#57606a">1,747 tokens</td>
    </tr>
  </table>
</div>


    
    Read result:
    Looking at the code, I can see that there's an issue with the `main.py` function signature and the `--help` documentation. The current implementation only accepts positional arguments for `input_path`, `output_path`, and an optional `schema_path`, but based on typical CLI tools and the test file naming (`test_cli.py`), it seems like this tool should support command-line flags like `--input`, `--ou...
    
    ↳ read



<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">Step 2/4  WRITE  Add type annotations to all functions in utils/formatter.py and update their docstrings</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>


    
    Applying: Add comprehensive type annotations to all function parameters and return values, and update docstrings to include parameter descriptions and return value descriptions following the existing docstring style.
    File: utils/formatter.py
    



<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">Proposed diff</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>



<pre style="background:#f6f8fa;border:1px solid #d0d7de;border-radius:6px;padding:12px;font-size:0.82rem;overflow-x:auto;line-height:1.4"><span style="color:#57606a">--- a/utils/formatter.py</span>
<span style="color:#57606a">+++ b/utils/formatter.py</span>
<span style="color:#0969da">@@ -10,6 +10,13 @@</span>
<span style="color:#57606a"> </span>
<span style="color:#57606a">     All keys across all records are used as headers.</span>
<span style="color:#57606a">     Missing values are rendered as empty strings.</span>
<span style="color:#2da44e">+</span>
<span style="color:#2da44e">+    Args:</span>
<span style="color:#2da44e">+        records: A list of dictionaries, each representing a record with flat key-value pairs.</span>
<span style="color:#2da44e">+        delimiter: The delimiter character to use in the CSV output. Defaults to ",".</span>
<span style="color:#2da44e">+</span>
<span style="color:#2da44e">+    Returns:</span>
<span style="color:#2da44e">+        A CSV-formatted string containing all records with headers derived from all unique keys.</span>
<span style="color:#57606a">     """</span>
<span style="color:#57606a">     if not records:</span>
<span style="color:#57606a">         return ""</span></pre>


    
    DRY RUN — file not written. Set dry_run=False to apply.
    ↳ wrote (dry)



<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">Step 3/4  WRITE  Add type annotations to all functions in utils/parser.py and update their docstrings</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>


    
    Applying: Add comprehensive type annotations to all function parameters and return values, and update docstrings to include parameter descriptions and return value descriptions following the existing docstring style.
    File: utils/parser.py
    


---
## Chapter 13 — Test Generation

**Goal:** the agent writes its own tests, runs them, and fixes failures.

This is the most complete demonstration of the full system.
Every function built across chapters 1–11 is in play:

- `run()` reads the source file to understand it (Ch 9)
- `write_file()` writes the test file to disk (Ch 10)
- `agent_loop()` orchestrates the plan (Ch 11)
- A subprocess call runs `pytest` and captures the output
- If tests fail, the failure output is fed back to `apply_patch()` and retried

**You will:**
- Write `generate_tests()` — ask the LLM to produce a pytest file for a module
- Write `run_tests()` — execute pytest, capture pass/fail/error
- Write `test_loop()` — generate → run → fix loop with a retry cap
- Try it: generate tests for `utils/formatter.py` and run them

### 13.1 `generate_tests()` — produce a pytest file

We read the source module first via `run()`, then ask the model
to write a complete `pytest` test file for it.
The test file is returned as a string — not written yet.


```python
def generate_tests(
    source_path: str,
    repo_root:   str = REPO_ROOT,
) -> str:
    """
    Generate a pytest test file for the module at *source_path*.

    Returns the test file content as a string (not written to disk yet).
    """
    full_path = Path(repo_root) / source_path
    source    = full_path.read_text(errors="ignore")

    prompt = (
        f"Write a complete pytest test file for the following Python module.\n\n"
        f"Requirements:\n"
        f"- Use pytest (not unittest)\n"
        f"- Cover the main happy path and at least two edge cases per function\n"
        f"- Import from the module using its dotted path relative to the repo root\n"
        f"  (e.g. 'from utils.formatter import to_csv')\n"
        f"- Return ONLY the test file — no explanation, no markdown fences\n\n"
        f"SOURCE FILE: {source_path}\n\n{source}"
    )

    raw, _ = chat([{"role": "user", "content": prompt}])

    # Strip fences if model added them
    code = re.sub(r"^```[a-zA-Z]*\n?", "", raw.strip())
    code = re.sub(r"\n?```$", "", code)
    return code.strip() + "\n"


# ── Preview without writing ───────────────────────────────────────────────────
show_rule("Generating tests for utils/formatter.py")
test_code = generate_tests("utils/formatter.py", repo_root="sample_project")
display(Markdown(f"```python\n{test_code}\n```"))

```


<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">Generating tests for utils/formatter.py</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>



```python
import pytest
from utils.formatter import to_csv


class TestToCsv:
    def test_empty_records(self):
        assert to_csv([]) == ""

    def test_single_record_basic(self):
        records = [{"name": "Alice", "age": "30"}]
        result = to_csv(records)
        assert result == "name,age\r\nAlice,30\r\n"

    def test_multiple_records_with_missing_values(self):
        records = [
            {"name": "Alice", "age": "30"},
            {"name": "Bob"},
            {"age": "25", "city": "Paris"}
        ]
        result = to_csv(records)
        lines = result.strip().split("\r\n")
        assert len(lines) == 4  # header + 3 records
        assert lines[0] == "name,age,city"
        assert lines[1] == "Alice,30,"
        assert lines[2] == "Bob,,"
        assert lines[3] == ",25,Paris"

    def test_custom_delimiter(self):
        records = [{"name": "Alice", "age": "30"}]
        result = to_csv(records, delimiter=";")
        assert result == "name;age\r\nAlice;30\r\n"

    def test_headers_order_preserved_first_seen(self):
        records = [
            {"z": "1", "a": "2"},
            {"b": "3", "a": "4"},
            {"c": "5"}
        ]
        result = to_csv(records)
        lines = result.strip().split("\r\n")
        assert lines[0] == "z,a,b,c"

    def test_records_with_none_values(self):
        records = [{"name": None, "age": "30"}]
        result = to_csv(records)
        assert result == "name,age\r\n,30\r\n"

    def test_records_with_numeric_values_as_strings(self):
        records = [{"value": 42}]
        result = to_csv(records)
        assert result == "value\r\n42\r\n"

    def test_records_with_special_characters_in_values(self):
        records = [{"name": "Alice, Jr.", "quote": "He said, \"Hello\""}]
        result = to_csv(records)
        # CSV should properly quote fields containing special characters
        assert result.startswith("name,quote\r\n")
        assert '"Alice, Jr."' in result or "Alice, Jr." in result  # both acceptable depending on CSV rules
        assert "He said, \"Hello\"" in result or '"He said, ""Hello"""' in result

    def test_large_number_of_records(self):
        records = [{"id": str(i), "value": i * 10} for i in range(1000)]
        result = to_csv(records)
        lines = result.strip().split("\r\n")
        assert len(lines) == 1001  # header + 1000 records
        assert lines[0] == "id,value"
        assert lines[1] == "0,0"
        assert lines[-1] == "999,9990"

```


### 13.2 `run_tests()` — execute pytest, capture results

We run `pytest` in a subprocess and return a structured result:
passed, failed, error count, and the raw output for feeding back
to the model if something went wrong.


```python
import subprocess as _sp


def run_tests(
    test_path: str,
    repo_root: str = REPO_ROOT,
) -> dict:
    """
    Run pytest on *test_path* (relative to repo_root).

    Returns:
        {
          "passed":  int,
          "failed":  int,
          "errors":  int,
          "output":  str,   # full pytest stdout+stderr
          "ok":      bool,  # True if passed > 0 and failed == errors == 0
        }
    """
    result = _sp.run(
        [sys.executable, "-m", "pytest", test_path, "-v", "--tb=short",
         "--no-header", "-q"],
        capture_output=True,
        text=True,
        cwd=repo_root,   # run from inside repo_root, so test_path is relative
    )

    output = result.stdout + result.stderr

    # Parse summary line: "3 passed, 1 failed, 0 errors"
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed",  output)) else 0
    failed = int(m.group(1)) if (m := re.search(r"(\d+) failed",  output)) else 0
    errors = int(m.group(1)) if (m := re.search(r"(\d+) error",   output)) else 0

    return {
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "output": output,
        "ok":     passed > 0 and failed == 0 and errors == 0,
    }
```

### 13.3 `test_loop()` — generate → run → fix → retry

The loop:
1. Generate a test file with `generate_tests()`
2. Run it with `run_tests()`
3. If tests fail, build a fix instruction from the **live pytest output** and rewrite the file
4. Repeat until tests pass or retries are exhausted

**Why not just use `agent_loop`?**

`agent_loop` plans upfront — it decides all steps before executing any of them. That works well when the task is predictable ("add type annotations to these three files"). It breaks down here because:

- The number of iterations is dynamic — we don't know in advance whether the generated tests will pass on the first try.
- Step 3's fix instruction depends on the *actual output* of step 2. That output doesn't exist at plan time.

`agent_loop` has no way to feed a step's result back into a later step's instruction. `test_loop` is a **feedback loop**, not a plan-then-execute sequence.

**What is reflection?**

A *reflecting* agent observes the result of each step before deciding the next one. After running pytest and seeing failures, it asks: *"Given what just happened, what should I do next?"* rather than following a fixed plan. This is sometimes called a ReAct loop (Reason + Act).

**Combining the strategies**

A unified loop would work like this:

```
plan → execute step → observe result → re-plan → execute step → ...
```

After each step, the agent appends the result to its context and asks the planner whether the remaining steps still make sense, or whether new steps are needed. `test_loop` is a hand-written specialisation of this pattern for the generate/test/fix cycle.

We will build this generalised reflection loop in a later chapter, at which point `test_loop` becomes a one-line call to `agent_loop` with reflection enabled.



```python
def test_loop(
    source_path: str,
    repo_root:   str = REPO_ROOT,
    max_retries: int = 3,
) -> dict:
    """
    Full generate → run → fix loop for *source_path*.

    Returns the final run_tests() result dict plus a "attempts" key.
    """
    # Derive test file path: utils/formatter.py → tests/test_formatter_gen.py
    stem      = Path(source_path).stem
    test_path = f"tests/test_{stem}_gen.py"
    full_test = Path(repo_root) / test_path
    full_test.parent.mkdir(parents=True, exist_ok=True)

    show_rule(f"Test Loop  ·  {source_path}")

    # ── 1. Generate initial tests ──────────────────────────────────────────────
    print("Step 1: generating tests…")
    test_code = generate_tests(source_path, repo_root=repo_root)
    full_test.write_text(test_code)
    print(f"Written: {test_path}")

    for attempt in range(1, max_retries + 2):   # +1 for initial attempt
        # ── 2. Run tests ──────────────────────────────────────────────────────
        show_rule(f"Attempt {attempt}")
        result = run_tests(test_path, repo_root=repo_root)

        color = "green" if result["ok"] else "red"
        icon = "✓" if result["ok"] else "✗"
        print(f"  {icon} passed={result['passed']}  failed={result['failed']}  errors={result['errors']}")

        if result["ok"]:
            print("\n✓ All tests pass.")
            result["attempts"] = attempt
            return result

        if attempt > max_retries:
            print(f"\nMax retries ({max_retries}) reached. Giving up.")
            break

        # ── 3. Fix ────────────────────────────────────────────────────────────
        print("\nTests failed. Asking model to fix…")
        print(result['output'][-1200:])   # last 1200 chars of pytest output

        fix_instruction = (
            f"The test file has failures. Fix ONLY the test code — do not modify the "
            f"source module. Here is the pytest output:\n\n{result['output'][-1500:]}"
        )
        _, fixed_code = apply_patch(test_path, fix_instruction, repo_root=repo_root)
        full_test.write_text(fixed_code)
        print(f"Rewritten: {test_path}")

    result["attempts"] = max_retries + 1
    return result
```

### 13.4 Try it — generate and run tests for `formatter.py`

This cell writes real files to `sample_project/tests/` and runs `pytest`.
Make sure `pytest` is installed (`pip install pytest`).


```python
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "pytest"],
    stdout=subprocess.DEVNULL,
)

result = test_loop(
    source_path = "utils/formatter.py",
    repo_root   = "sample_project",
    max_retries = 3,
)

icon = "✓ PASS" if result["ok"] else "✗ FAIL"
print(f"\nFinal result: {icon}  in {result['attempts']} attempt(s)")
```


<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">Test Loop  ·  utils/formatter.py</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>


    Step 1: generating tests…
    Written: tests/test_formatter_gen.py



<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">Attempt 1</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>


      ✗ passed=1  failed=6  errors=0
    
    Tests failed. Asking model to fix…
     id,name,active[39;49;00m[90m[39;49;00m[0m
    [1m[31mE     [92m+ id,name,active[39;49;00m[90m[39;49;00m[0m
    [1m[31mE     [90m?               +[39;49;00m[0m
    [36m[1m=========================== short test summary info ============================[0m
    [31mFAILED[0m tests/test_formatter_gen.py::[1mtest_to_csv_basic[0m - AssertionError: assert 'name,age,city\r' == 'name,age,city'
    [31mFAILED[0m tests/test_formatter_gen.py::[1mtest_to_csv_single_record[0m - AssertionError: assert 'name,age\r' == 'name,age'
    [31mFAILED[0m tests/test_formatter_gen.py::[1mtest_to_csv_missing_values[0m - AssertionError: assert 'name,city,age\r' == 'name,city,age'
    [31mFAILED[0m tests/test_formatter_gen.py::[1mtest_to_csv_custom_delimiter[0m - AssertionError: assert 'name;age\r' == 'name;age'
    [31mFAILED[0m tests/test_formatter_gen.py::[1mtest_to_csv_key_order_preserved[0m - AssertionError: assert 'z,a,m\r' == 'z,a,m'
    [31mFAILED[0m tests/test_formatter_gen.py::[1mtest_to_csv_mixed_types[0m - AssertionError: assert 'id,name,active\r' == 'id,name,active'
    [31m========================= [31m[1m6 failed[0m, [32m1 passed[0m[31m in 0.04s[0m[31m ==========================[0m
    
    Rewritten: tests/test_formatter_gen.py



<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">Attempt 2</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>


      ✓ passed=7  failed=0  errors=0
    
    ✓ All tests pass.
    
    Final result: ✓ PASS  in 2 attempt(s)


---

## Chapter 14 — Adding a New Capability

The agent is a pipeline of functions connected by a small number of **switch points** — places where the code checks an action type or strategy name and branches. Adding a new capability always means the same three things:

1. **Write the function** — pure Python, no magic.
2. **Wire it in** — add a case to the right switch point.
3. **Tell the planner** — add a line to `_PLAN_SYSTEM` so the LLM knows the action exists.

That's the whole pattern. Everything downstream — the agent loop, the Streamlit UI, the activity log — picks it up automatically.

### The three switch points

| What you're adding | Switch point |
|---|---|
| New retrieval strategy | `pick_strategy()` + new `*_retrieve()` function |
| New plan action | `_PLAN_SYSTEM` prompt + `plan_task()` parser + `execute_step()` |
| New file types to index/embed | `CODE_EXTENSIONS` set |

### 14.1 The worked example: `ACTION: fetch`

We'll add the ability for the agent to fetch a URL — useful when the task says "implement this API" and you paste in a link to the docs.

The plan action will look like:

```
STEP: Read the requests library docs
ACTION: fetch
URL: https://docs.python-requests.org/en/latest/
```

Three changes required. We'll do them one at a time.


### 14.1 — Write the function

`fetch_url()` downloads a page and strips HTML tags so the model gets readable text rather than a wall of `<div>` soup. Nothing agent-specific here — just plain Python.



```python
import re as _re

def fetch_url(url: str) -> dict:
    """
    Fetch *url* and return plain text (HTML tags stripped).

    Returns {"url": str, "text": str, "ok": bool, "error": str | None}
    """
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "pocket-agent/1.0"})
        resp.raise_for_status()
        # Strip every HTML tag with a simple regex — good enough for docs pages
        text = _re.sub(r"<[^>]+>", " ", resp.text)
        # Collapse whitespace
        text = _re.sub(r"\s{2,}", "\n", text).strip()
        return {"url": url, "text": text, "ok": True, "error": None}
    except Exception as exc:
        return {"url": url, "text": "", "ok": False, "error": str(exc)}

# Quick smoke-test
result = fetch_url("https://httpbin.org/html")
print(f"ok={result['ok']}  chars={len(result['text'])}")
print(result["text"][:300])

```

    ok=True  chars=3594
    Herman Melville - Moby-Dick
    Availing himself of the mild, summer-cool weather that now reigned in these latitudes, and in preparation for the peculiarly active pursuits shortly to be anticipated, Perth, the begrimed, blistered old blacksmith, had not removed his portable forge to the hold again, aft


### 14.2 — Wire it into `execute_step()`

`execute_step()` is the second switch point. It already handles `read`, `write`, `bash`, `glob`, and `grep`. We add `fetch` and `glob`/`grep` as first-class branches — each returning the same structured dict shape the rest of the loop expects.

Note that the fetched text is also passed to the LLM as context — so the agent can answer questions *about* the page, not just confirm it was downloaded.



```python
# Chapter 14 redefinition of execute_step — three changes from Chapter 12:
#   1. Returns a structured dict instead of a plain string — the caller
#      (agent_loop) decides how to display the result.
#   2. Removes dry_run — write steps always produce a diff for review;
#      committing to disk is a separate decision.
#   3. Adds ACTION: fetch.

def execute_step(
    step:      dict,
    repo_root: str = REPO_ROOT,
) -> dict:
    """
    Execute one plan step. Returns a structured result dict.

      read  → {"type": "read",  "output": str}
      write → {"type": "write", "file_path": str, "new_content": str, "diff": str}
      bash  → {"type": "bash",  "cmd": str, "output": str, "ok": bool}
      glob  → {"type": "glob",  "pattern": str, "files": list[str]}
      grep  → {"type": "grep",  "pattern": str, "matches": list[str]}
      fetch → {"type": "fetch", "url": str, "output": str, "ok": bool}
    """
    action = step.get("action", "read")

    # ── fetch ─────────────────────────────────────────────────────────────────
    if action == "fetch":
        url    = step.get("url", step.get("target", ""))
        result = fetch_url(url)
        if not result["ok"]:
            return {"type": "fetch", "url": url, "output": f"Error: {result['error']}", "ok": False}
        summary_reply, _ = chat([
            {"role": "system", "content": "Summarise the following web page concisely."},
            {"role": "user",   "content": f"URL: {url}\n\n{result['text']}"},
        ])
        return {"type": "fetch", "url": url, "output": summary_reply, "ok": True}

    # ── bash ──────────────────────────────────────────────────────────────────
    if action == "bash":
        cmd  = step.get("cmd", "")
        proc = tool_bash(cmd, repo_root=repo_root)
        return {"type": "bash", "cmd": cmd,
                "output": (proc["stdout"] + proc["stderr"]).strip(),
                "ok": proc["code"] == 0}

    # ── glob ──────────────────────────────────────────────────────────────────
    if action == "glob":
        pattern = step.get("pattern", "**/*")
        files   = tool_glob(pattern, repo_root=repo_root)
        return {"type": "glob", "pattern": pattern,
                "files": [f["path"] for f in files]}

    # ── grep ──────────────────────────────────────────────────────────────────
    if action == "grep":
        pattern = step.get("pattern", "")
        matches = tool_grep(pattern, repo_root=repo_root)
        return {"type": "grep", "pattern": pattern,
                "matches": [f["path"] for f in matches]}

    # ── write ─────────────────────────────────────────────────────────────────
    if action == "write":
        original, proposed, diff = write_file(
            file_path=step["target"], instruction=step["instruction"],
            repo_root=repo_root, dry_run=True,
        )
        return {"type": "write", "file_path": step["target"],
                "new_content": proposed, "diff": diff}

    # ── read (default) ────────────────────────────────────────────────────────
    result = run(step["target"], repo_root=repo_root)
    return {"type": "read", "output": result.reply}
```


```python
# Chapter 14 redefinition of agent_loop — changes from Chapter 12:
#   1. Removes dry_run and max_steps
#   2. Handles structured dict results from execute_step()
#   3. Shows diffs and asks for confirmation before writing
#   4. Verifies written files exist on disk

def agent_loop(
    task:      str,
    repo_root: str = REPO_ROOT,
) -> list[dict]:
    """
    Autonomous task execution loop.

    1. Plan the task with plan_task()
    2. Execute each step with execute_step() — returns structured dicts
    3. For write steps: show diff, ask confirmation, write on accept
    4. Verify all accepted writes exist on disk

    Returns the step log (list of dicts with "step", "action", "result", "accepted").
    """
    show_rule(f"Agent Loop  ·  {task[:60]}")
    print(f"repo: {repo_root}\n")

    # ── 1. Plan ───────────────────────────────────────────────────────────────
    plan = plan_task(task, repo_root=repo_root)
    if not plan:
        print("Planning failed — no steps parsed from model output.")
        return []

    print(f"Plan: {len(plan)} step(s)\n")

    # ── 2. Execute ────────────────────────────────────────────────────────────
    log = []
    for i, step in enumerate(plan, 1):
        show_rule(f"Step {i}/{len(plan)}  {step['action'].upper()}  {step['step']}")
        result   = execute_step(step, repo_root=repo_root)
        accepted = False

        if result["type"] == "write":
            print(result["diff"])
            answer = input("\nAccept this write? [y/N] ").strip().lower()
            if answer == "y":
                path = Path(repo_root) / result["file_path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(result["new_content"])
                print(f"Written → {result['file_path']}")
                accepted = True
            else:
                print("Skipped.")

        elif result["type"] == "fetch":
            icon = "✅" if result["ok"] else "❌"
            print(f"{icon} fetched {result['url']}")
            show_reply(result["output"])

        elif result["type"] == "bash":
            icon = "✅" if result["ok"] else "❌"
            print(f"{icon} `{result['cmd']}`")
            if result["output"]:
                print(result["output"][:400])

        elif result["type"] in ("glob", "grep"):
            items = result.get("files") or result.get("matches") or []
            print(f"{len(items)} result(s): " + ", ".join(items[:10]))

        else:  # read
            show_reply(result.get("output", ""))

        log.append({**step, "result": result, "accepted": accepted})

    # ── 3. Verify ─────────────────────────────────────────────────────────────
    accepted_writes = [s for s in log if s["result"]["type"] == "write" and s["accepted"]]
    if accepted_writes:
        show_rule("Verification")
        all_ok = True
        for s in accepted_writes:
            path = Path(repo_root) / s["result"]["file_path"]
            icon = "✅" if path.exists() else "❌"
            print(f"{icon} {s['result']['file_path']}")
            if not path.exists():
                all_ok = False
        print("\nAll files confirmed on disk." if all_ok else "\nSome files are missing.")

    show_rule("Loop complete")
    return log

```

### 14.3 — Tell the planner

We also add `URL:` to the parser's regex so the field is captured correctly.



```python
# Chapter 14 extends _PLAN_SYSTEM to add ACTION: fetch.
# Re-running this cell replaces the Chapter 12 version in memory.

_PLAN_SYSTEM = """\
You are a coding agent. Given a task and a project map, produce an ordered
list of ALL steps needed to fully complete the task.

Each step must be one of these formats (blank line between steps):

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
ACTION: glob
PATTERN: <glob pattern, e.g. utils/**/*.py>

STEP: <one-sentence description>
ACTION: grep
PATTERN: <regex pattern to search file contents>

STEP: <one-sentence description>
ACTION: fetch
URL: <full URL to retrieve>

Rules:
- Use ACTION: fetch — never ACTION: bash with curl or wget — when the task references
  an external URL or documentation link.
- Prefer ACTION: bash for file system operations (copy, move, delete, rename).
- Use ACTION: glob to find files by name pattern.
- Use ACTION: grep to search file contents for a regex pattern.
- Use ACTION: read only when you need code understanding before you can write.
- Use ACTION: write when the LLM must generate or edit file content.
- Never invent file paths; never use absolute or /tmp paths.
- A single READ step retrieves from the whole codebase — ask about multiple files
  in one step rather than one step per file.
- Each WRITE step targets exactly one file.
- Be specific in INSTRUCTION — the change will be applied by another model with no extra context.
- Do not add any text outside the step blocks.
"""

```


```python
# Chapter 14 redefinition of plan_task — extends Chapter 12 to add fetch.

def plan_task(task: str, repo_root: str = REPO_ROOT) -> list[dict]:
    """
    Ask the LLM to produce a step-by-step plan for *task*.

    Extends the Chapter 12 version with one new action type:
        fetch  — retrieve an external URL

    Returns a list of dicts:
        {"step": str, "action": "read"|"write"|"bash"|"glob"|"grep"|"fetch",
         "target": str, "instruction": str, "cmd": str, "pattern": str, "url": str}
    """
    manifest  = load_manifest(repo_root)
    all_files = glob_files("*", repo_root=repo_root)
    file_list = "\n".join(f["path"] for f in all_files)

    prompt = (
        f"PROJECT MAP:\n{manifest['text']}\n\n"
        f"AVAILABLE FILES (use only if the task involves the codebase):\n{file_list}\n\n"
        f"TASK: {task}"
    )

    raw, _ = chat([
        {"role": "system", "content": _PLAN_SYSTEM},
        {"role": "user",   "content": prompt},
    ])

    print(raw)

    steps = []
    for block in re.split(r"(?m)(?=^STEP:)", raw.strip()):
        if not block.strip():
            continue
        lines = {
            m.group(1).lower(): m.group(2).strip()
            for line in block.splitlines()
            if (m := re.match(
                r"^(STEP|ACTION|TARGET|INSTRUCTION|CMD|PATTERN|URL):\s*(.+)$",
                line, re.IGNORECASE
            ))
        }
        action = lines.get("action", "").lower()
        has_content = (
            "target"  in lines or
            "cmd"     in lines or
            "pattern" in lines or
            "url"     in lines or
            action in ("read", "write", "bash", "glob", "grep", "fetch")
        )
        if action and has_content:
            steps.append({
                "step":        lines.get("step", ""),
                "action":      action,
                "target":      lines.get("target", ""),
                "instruction": lines.get("instruction", ""),
                "cmd":         lines.get("cmd", ""),
                "pattern":     lines.get("pattern", ""),
                "url":         lines.get("url", ""),
            })

    return steps

```

### 14.4 Try it — run the agent with a URL task

The cell below asks the agent to fetch the Python `pathlib` docs and write a one-page cheat sheet based on them. The plan should come back with a `fetch` step followed by a `write` step — no reads needed.



```python
log = agent_loop(
    task      = ("Fetch https://docs.python.org/3/library/pathlib.html "
                 "and write a concise cheat sheet to docs/pathlib_cheatsheet.md"),
    repo_root = REPO_ROOT,
)

```


<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">Agent Loop  ·  Fetch https://docs.python.org/3/library/pathlib.html and wri</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>


    repo: .
    
    STEP: Fetch the Python pathlib documentation
    ACTION: fetch
    URL: https://docs.python.org/3/library/pathlib.html
    
    STEP: Write a concise cheat sheet to docs/pathlib_cheatsheet.md
    ACTION: write
    TARGET: docs/pathlib_cheatsheet.md
    INSTRUCTION: Write a concise, practical cheat sheet for Python's pathlib module based on the documentation at https://docs.python.org/3/library/pathlib.html. Include: (1) common import patterns, (2) core classes (Path vs PurePath), (3) essential methods for path operations (creating, joining, resolving, checking existence), (4) path components accessors (parent, name, suffix, stem), (5) file operations (read_text, read_bytes, write_text, write_bytes, exists, is_file, is_dir), (6) directory operations (glob, rglob, iterdir, mkdir), (7) path formatting with examples. Use a clean markdown format with short code snippets for each section. Keep it under 200 lines.
    Plan: 2 step(s)
    



<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">Step 1/2  FETCH  Fetch the Python pathlib documentation</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>


    ✅ fetched https://docs.python.org/3/library/pathlib.html



The **`pathlib`** module in Python provides an object-oriented interface for filesystem paths, introduced in Python 3.4. It centralizes path operations in classes—**`PurePath`** for computational (non-I/O) operations and **`Path`** (a concrete subclass) for I/O operations like file access and manipulation.

### Key Concepts:
- **Pure Paths** (`PurePath`, `PurePosixPath`, `PureWindowsPath`):  
  Platform-agnostic, no filesystem interaction. Useful for cross-platform path manipulation.
- **Concrete Paths** (`Path`, `PosixPath`, `WindowsPath`):  
  Perform actual filesystem operations. `Path` instantiates the platform-specific concrete class (`PosixPath` on Unix, `WindowsPath` on Windows).

### Core Functionality:
- **Path Creation & Joining**:  
  Use `Path(...)`, `/` operator, or `joinpath()` to construct paths (e.g., `Path("/etc") / "hosts"`).
- **Path Components**:  
  Access parts via `parts`, `drive`, `root`, `anchor`, `parents`, `parent`, `name`, `stem`, `suffix`, `suffixes`.
- **Absolute/Relative Paths**:  
  `absolute()`, `resolve()` (resolves symlinks), `is_absolute()`, `relative_to()`.
- **Filesystem Operations**:
  - **Query**: `exists()`, `is_file()`, `is_dir()`, `is_symlink()`, `is_mount()`, `stat()`, `lstat()`, `samefile()`.
  - **Read/Write**: `open()`, `read_text()`/`read_bytes()`, `write_text()`/`write_bytes()`.
  - **Directory Access**: `iterdir()`, `glob()`, `rglob()`, `walk()`.
  - **Creation/Modification**: `mkdir()`, `touch()`, `symlink_to()`, `hardlink_to()`.
  - **Copy/Move/Delete**: `copy()`, `move()`, `rename()`, `replace()`, `unlink()`, `rmdir()`.
  - **Permissions/Owner**: `chmod()`, `owner()`, `group()`.

### Features & Notes:
- **Path Operations Are Immutable & Hashable**.
- **Pattern Matching**: Supports glob patterns (`*`, `?`, `[...]`, `**`) for `glob()`, `rglob()`, and `full_match()`.
- **URI Handling**: `from_uri()`/`as_uri()` support RFC 8089 file URIs (added in 3.13).
- **Symlinks**: Respects symlinks by default; use `follow_symlinks=False` for lstat-like behavior.
- **Error Handling**: Raises `UnsupportedOperation` for unsupported operations (e.g., `WindowsPath` on Unix), and `OSError` for filesystem issues.

### Comparison to `os.path`:
- `pathlib` uses objects instead of strings (more readable, object-oriented).
- Does *not* support bytes paths or directory descriptors.
- More opinionated normalization (e.g., `..` preserved in `absolute()`).
- Not a drop-in replacement for `os.path` due to behavioral differences.

### Added in Version 3.14:
- `PathInfo` caching protocol, `full_match()`, trailing-slash globbing, single-dot suffix support, and more.

For low-level or performance-critical string-based operations, `os.path` may be preferred. For most modern use cases, `pathlib` offers a cleaner, safer API.



<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">Step 2/2  WRITE  Write a concise cheat sheet to docs/pathlib_cheatsheet.md</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>


    
    Applying: Write a concise, practical cheat sheet for Python's pathlib module based on the documentation at https://docs.python.org/3/library/pathlib.html. Include: (1) common import patterns, (2) core classes (Path vs PurePath), (3) essential methods for path operations (creating, joining, resolving, checking existence), (4) path components accessors (parent, name, suffix, stem), (5) file operations (read_text, read_bytes, write_text, write_bytes, exists, is_file, is_dir), (6) directory operations (glob, rglob, iterdir, mkdir), (7) path formatting with examples. Use a clean markdown format with short code snippets for each section. Keep it under 200 lines.
    File: docs/pathlib_cheatsheet.md
    



<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">Proposed diff</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>



<pre style="background:#f6f8fa;border:1px solid #d0d7de;border-radius:6px;padding:12px;font-size:0.82rem;overflow-x:auto;line-height:1.4"><span style="color:#57606a">--- a/docs/pathlib_cheatsheet.md</span>
<span style="color:#57606a">+++ b/docs/pathlib_cheatsheet.md</span>
<span style="color:#0969da">@@ -5,94 +5,77 @@</span>
<span style="color:#57606a"> from pathlib import Path, PurePath, PureWindowsPath, PurePosixPath</span>
<span style="color:#57606a"> ```</span>
<span style="color:#57606a"> </span>
<span style="color:#cf222e">-## Core Concepts</span>
<span style="color:#cf222e">-- **Path objects** are immutable (except when using certain OS operations)</span>
<span style="color:#cf222e">-- **Windows vs POSIX**: Use `PureWindowsPath`, `PurePosixPath`, or generic `Path` (auto-detects OS)</span>
<span style="color:#2da44e">+## Core Classes</span>
<span style="color:#2da44e">+- `Path`: OS-dependent, supports filesystem operations  </span>
<span style="color:#2da44e">+- `PurePath`: OS-agnostic, *no* filesystem operations  </span>
<span style="color:#2da44e">+- `PureWindowsPath`/`PurePosixPath`: Explicitly target OS</span>
<span style="color:#57606a"> </span>
<span style="color:#57606a"> ---</span>
<span style="color:#57606a"> </span>
<span style="color:#cf222e">-## Creating Paths</span>
<span style="color:#cf222e">-</span>
<span style="color:#cf222e">-| Operation | Code |</span>
<span style="color:#cf222e">-|-----------|------|</span>
<span style="color:#cf222e">-| Current directory | `Path()` or `Path('.')` |</span>
<span style="color:#cf222e">-| Home directory | `Path.home()` |</span>
<span style="color:#cf222e">-| Working directory | `Path.cwd()` |</span>
<span style="color:#cf222e">-| Join paths | `Path('/tmp') / 'file.txt'` or `Path('/tmp').joinpath('dir', 'file.txt')` |</span>
<span style="color:#cf222e">-| From string | `Path('C:\\Users\\file.txt')` (Windows) or `Path('/home/file.txt')` (Linux/macOS) |</span>
<span style="color:#2da44e">+## Path Creation</span>
<span style="color:#2da44e">+```python</span>
<span style="color:#2da44e">+Path()                      # current dir</span>
<span style="color:#2da44e">+Path.home()                 # user’s home</span>
<span style="color:#2da44e">+Path.cwd()                  # current working dir</span>
<span style="color:#2da44e">+Path('/tmp') / 'file.txt'   # join with /</span>
<span style="color:#2da44e">+Path('/tmp').joinpath('dir', 'file.txt')  # joinpath</span>
<span style="color:#2da44e">+Path(r'C:\file.txt')        # Windows (raw string safe)</span>
<span style="color:#2da44e">+```</span>
<span style="color:#57606a"> </span>
<span style="color:#57606a"> ---</span>
<span style="color:#57606a"> </span>
<span style="color:#cf222e">-## Path Attributes &amp; Methods</span>
<span style="color:#cf222e">-</span>
<span style="color:#cf222e">-| Method | Description | Example |</span>
<span style="color:#cf222e">-|--------|-------------|---------|</span>
<span style="color:#cf222e">-| `str(p)` | Convert to string | `str(Path('/tmp/file.txt'))` → `'/tmp/file.txt'` |</span>
<span style="color:#cf222e">-| `p.name` | Final path component | `Path('/tmp/file.txt').name` → `'file.txt'` |</span>
<span style="color:#cf222e">-| `p.stem` | Name without extension | `Path('file.tar.gz').stem` → `'file.tar'` |</span>
<span style="color:#cf222e">-| `p.suffix` | Last extension | `Path('file.tar.gz').suffix` → `'.gz'` |</span>
<span style="color:#cf222e">-| `p.suffixes` | All extensions | `Path('file.tar.gz').suffixes` → `['.tar', '.gz']` |</span>
<span style="color:#cf222e">-| `p.parent` | Parent directory | `Path('/tmp/file.txt').parent` → `PosixPath('/tmp')` |</span>
<span style="color:#cf222e">-| `p.parents` | Ancestors (list-like) | `Path('/a/b/c').parents[0]` → `PosixPath('/a/b')` |</span>
<span style="color:#cf222e">-| `p.parts` | Tuple of components | `Path('/a/b/c').parts` → `('/', 'a', 'b', 'c')` |</span>
<span style="color:#2da44e">+## Path Components</span>
<span style="color:#2da44e">+| Attribute | Example | Result |</span>
<span style="color:#2da44e">+|-----------|---------|--------|</span>
<span style="color:#2da44e">+| `p.name` | `Path('a/b/file.txt').name` | `'file.txt'` |</span>
<span style="color:#2da44e">+| `p.stem` | `Path('file.tar.gz').stem` | `'file.tar'` |</span>
<span style="color:#2da44e">+| `p.suffix` | `Path('file.tar.gz').suffix` | `'.gz'` |</span>
<span style="color:#2da44e">+| `p.suffixes` | `Path('file.tar.gz').suffixes` | `['.tar', '.gz']` |</span>
<span style="color:#2da44e">+| `p.parent` | `Path('a/b/c').parent` | `PosixPath('a/b')` |</span>
<span style="color:#2da44e">+| `p.parents` | `Path('a/b/c').parents[1]` | `PosixPath('a')` |</span>
<span style="color:#2da44e">+| `p.parts` | `Path('/a/b').parts` | `('/', 'a', 'b')` |</span>
<span style="color:#57606a"> </span>
<span style="color:#57606a"> ---</span>
<span style="color:#57606a"> </span>
<span style="color:#57606a"> ## Path Operations</span>
<span style="color:#cf222e">-</span>
<span style="color:#cf222e">-| Operation | Code |</span>
<span style="color:#cf222e">-|-----------|------|</span>
<span style="color:#cf222e">-| Check existence | `p.exists()` |</span>
<span style="color:#cf222e">-| Check is file | `p.is_file()` |</span>
<span style="color:#cf222e">-| Check is dir | `p.is_dir()` |</span>
<span style="color:#cf222e">-| Absolute path | `p.absolute()` |</span>
<span style="color:#cf222e">-| Resolve symlink | `p.resolve(strict=False)` |</span>
<span style="color:#cf222e">-| Relative to | `p.relative_to(Path('/tmp'))` |</span>
<span style="color:#cf222e">-| Normalize | `p.normpath()` → use `PurePath.normpath()` on `PurePath` instances |</span>
<span style="color:#2da44e">+```python</span>
<span style="color:#2da44e">+p.resolve()                # absolute, resolve symlinks</span>
<span style="color:#2da44e">+p.absolute()               # absolute (no symlink resolution)</span>
<span style="color:#2da44e">+p.exists()                 # path exists?</span>
<span style="color:#2da44e">+p.is_file()                # is file?</span>
<span style="color:#2da44e">+p.is_dir()                 # is directory?</span>
<span style="color:#2da44e">+p.relative_to(Path('/a'))  # relative path from /a</span>
<span style="color:#2da44e">+PurePath('a/b').normpath() # normalize (PurePath only)</span>
<span style="color:#2da44e">+```</span>
<span style="color:#57606a"> </span>
<span style="color:#57606a"> ---</span>
<span style="color:#57606a"> </span>
<span style="color:#cf222e">-## Filesystem Operations (on `Path` only)</span>
<span style="color:#cf222e">-</span>
<span style="color:#cf222e">-| Action | Method |</span>
<span style="color:#cf222e">-|--------|--------|</span>
<span style="color:#cf222e">-| Read text | `p.read_text(encoding='utf-8')` |</span>
<span style="color:#cf222e">-| Write text | `p.write_text('data', encoding='utf-8')` |</span>
<span style="color:#cf222e">-| Read bytes | `p.read_bytes()` |</span>
<span style="color:#cf222e">-| Write bytes | `p.write_bytes(b'data')` |</span>
<span style="color:#cf222e">-| Create dir | `p.mkdir(parents=True, exist_ok=True)` |</span>
<span style="color:#cf222e">-| Remove file | `p.unlink(missing_ok=False)` |</span>
<span style="color:#cf222e">-| Remove dir | `p.rmdir()` (empty only) |</span>
<span style="color:#cf222e">-| Remove tree | `shutil.rmtree(p)` (standard lib) |</span>
<span style="color:#cf222e">-| Copy file | `shutil.copy(p, dest)` |</span>
<span style="color:#cf222e">-| Move | `shutil.move(p, dest)` |</span>
<span style="color:#cf222e">-| Glob patterns | `list(Path('/tmp').glob('*.txt'))` |</span>
<span style="color:#cf222e">-| Recursive glob | `list(Path('/tmp').rglob('*.txt'))` |</span>
<span style="color:#2da44e">+## File Operations</span>
<span style="color:#2da44e">+```python</span>
<span style="color:#2da44e">+p.read_text(encoding='utf-8')     # read file as string</span>
<span style="color:#2da44e">+p.read_bytes()                    # read as bytes</span>
<span style="color:#2da44e">+p.write_text('data', 'utf-8')     # write string</span>
<span style="color:#2da44e">+p.write_bytes(b'data')            # write bytes</span>
<span style="color:#2da44e">+p.exists(), p.is_file(), p.is_dir()  # checks</span>
<span style="color:#2da44e">+```</span>
<span style="color:#57606a"> </span>
<span style="color:#57606a"> ---</span>
<span style="color:#57606a"> </span>
<span style="color:#cf222e">-## Common Patterns</span>
<span style="color:#2da44e">+## Directory Operations</span>
<span style="color:#2da44e">+```python</span>
<span style="color:#2da44e">+p.mkdir(parents=True, exist_ok=True)  # create dir + parents</span>
<span style="color:#2da44e">+p.iterdir()            # iterate direct children</span>
<span style="color:#2da44e">+list(p.glob('*.py'))   # glob pattern (non-recursive)</span>
<span style="color:#2da44e">+list(p.rglob('**/*.py'))  # recursive glob</span>
<span style="color:#2da44e">+```</span>
<span style="color:#57606a"> </span>
<span style="color:#cf222e">-- **Iterate over files in dir**:</span>
<span style="color:#cf222e">-  ```python</span>
<span style="color:#cf222e">-  for f in Path('/tmp').iterdir(): ...</span>
<span style="color:#cf222e">-  ```</span>
<span style="color:#2da44e">+---</span>
<span style="color:#57606a"> </span>
<span style="color:#cf222e">-- **Create parent directories if missing**:</span>
<span style="color:#cf222e">-  ```python</span>
<span style="color:#cf222e">-  (Path('a/b/c') / 'file.txt').parent.mkdir(parents=True, exist_ok=True)</span>
<span style="color:#cf222e">-  ```</span>
<span style="color:#cf222e">-</span>
<span style="color:#cf222e">-- **Read config safely**:</span>
<span style="color:#cf222e">-  ```python</span>
<span style="color:#cf222e">-  if (p := Path('config.json')).exists(): data = p.read_text()</span>
<span style="color:#cf222e">-  ```</span>
<span style="color:#cf222e">-</span>
<span style="color:#cf222e">-- **Find newest file**:</span>
<span style="color:#cf222e">-  ```python</span>
<span style="color:#cf222e">-  max(Path('/tmp').glob('*.log'), key=lambda x: x.stat().st_mtime)</span>
<span style="color:#cf222e">-  ```</span>
<span style="color:#cf222e">-</span>
<span style="color:#cf222e">-- **Check permissions**:</span>
<span style="color:#cf222e">-  ```python</span>
<span style="color:#cf222e">-  p.access(os.R_OK)  # requires import os</span>
<span style="color:#2da44e">+## Formatting Examples</span>
<span style="color:#2da44e">+```python</span>
<span style="color:#2da44e">+p = Path('/home/user/docs')</span>
<span style="color:#2da44e">+str(p)           # '/home/user/docs'</span>
<span style="color:#2da44e">+p.as_posix()     # '/home/user/docs' (always forward slashes)</span>
<span style="color:#2da44e">+p.with_name('new.txt')     # '/home/user/new.txt'</span>
<span style="color:#2da44e">+p.with_suffix('.json')     # '/home/user/docs.json'</span>
<span style="color:#2da44e">+p / 'subdir' / 'file.txt'  # /home/user/docs/subdir/file.txt</span></pre>


    
    DRY RUN — file not written. Set dry_run=False to apply.
    --- a/docs/pathlib_cheatsheet.md
    +++ b/docs/pathlib_cheatsheet.md
    @@ -5,94 +5,77 @@
     from pathlib import Path, PurePath, PureWindowsPath, PurePosixPath
     ```
     
    -## Core Concepts
    -- **Path objects** are immutable (except when using certain OS operations)
    -- **Windows vs POSIX**: Use `PureWindowsPath`, `PurePosixPath`, or generic `Path` (auto-detects OS)
    +## Core Classes
    +- `Path`: OS-dependent, supports filesystem operations  
    +- `PurePath`: OS-agnostic, *no* filesystem operations  
    +- `PureWindowsPath`/`PurePosixPath`: Explicitly target OS
     
     ---
     
    -## Creating Paths
    -
    -| Operation | Code |
    -|-----------|------|
    -| Current directory | `Path()` or `Path('.')` |
    -| Home directory | `Path.home()` |
    -| Working directory | `Path.cwd()` |
    -| Join paths | `Path('/tmp') / 'file.txt'` or `Path('/tmp').joinpath('dir', 'file.txt')` |
    -| From string | `Path('C:\\Users\\file.txt')` (Windows) or `Path('/home/file.txt')` (Linux/macOS) |
    +## Path Creation
    +```python
    +Path()                      # current dir
    +Path.home()                 # user’s home
    +Path.cwd()                  # current working dir
    +Path('/tmp') / 'file.txt'   # join with /
    +Path('/tmp').joinpath('dir', 'file.txt')  # joinpath
    +Path(r'C:\file.txt')        # Windows (raw string safe)
    +```
     
     ---
     
    -## Path Attributes & Methods
    -
    -| Method | Description | Example |
    -|--------|-------------|---------|
    -| `str(p)` | Convert to string | `str(Path('/tmp/file.txt'))` → `'/tmp/file.txt'` |
    -| `p.name` | Final path component | `Path('/tmp/file.txt').name` → `'file.txt'` |
    -| `p.stem` | Name without extension | `Path('file.tar.gz').stem` → `'file.tar'` |
    -| `p.suffix` | Last extension | `Path('file.tar.gz').suffix` → `'.gz'` |
    -| `p.suffixes` | All extensions | `Path('file.tar.gz').suffixes` → `['.tar', '.gz']` |
    -| `p.parent` | Parent directory | `Path('/tmp/file.txt').parent` → `PosixPath('/tmp')` |
    -| `p.parents` | Ancestors (list-like) | `Path('/a/b/c').parents[0]` → `PosixPath('/a/b')` |
    -| `p.parts` | Tuple of components | `Path('/a/b/c').parts` → `('/', 'a', 'b', 'c')` |
    +## Path Components
    +| Attribute | Example | Result |
    +|-----------|---------|--------|
    +| `p.name` | `Path('a/b/file.txt').name` | `'file.txt'` |
    +| `p.stem` | `Path('file.tar.gz').stem` | `'file.tar'` |
    +| `p.suffix` | `Path('file.tar.gz').suffix` | `'.gz'` |
    +| `p.suffixes` | `Path('file.tar.gz').suffixes` | `['.tar', '.gz']` |
    +| `p.parent` | `Path('a/b/c').parent` | `PosixPath('a/b')` |
    +| `p.parents` | `Path('a/b/c').parents[1]` | `PosixPath('a')` |
    +| `p.parts` | `Path('/a/b').parts` | `('/', 'a', 'b')` |
     
     ---
     
     ## Path Operations
    -
    -| Operation | Code |
    -|-----------|------|
    -| Check existence | `p.exists()` |
    -| Check is file | `p.is_file()` |
    -| Check is dir | `p.is_dir()` |
    -| Absolute path | `p.absolute()` |
    -| Resolve symlink | `p.resolve(strict=False)` |
    -| Relative to | `p.relative_to(Path('/tmp'))` |
    -| Normalize | `p.normpath()` → use `PurePath.normpath()` on `PurePath` instances |
    +```python
    +p.resolve()                # absolute, resolve symlinks
    +p.absolute()               # absolute (no symlink resolution)
    +p.exists()                 # path exists?
    +p.is_file()                # is file?
    +p.is_dir()                 # is directory?
    +p.relative_to(Path('/a'))  # relative path from /a
    +PurePath('a/b').normpath() # normalize (PurePath only)
    +```
     
     ---
     
    -## Filesystem Operations (on `Path` only)
    -
    -| Action | Method |
    -|--------|--------|
    -| Read text | `p.read_text(encoding='utf-8')` |
    -| Write text | `p.write_text('data', encoding='utf-8')` |
    -| Read bytes | `p.read_bytes()` |
    -| Write bytes | `p.write_bytes(b'data')` |
    -| Create dir | `p.mkdir(parents=True, exist_ok=True)` |
    -| Remove file | `p.unlink(missing_ok=False)` |
    -| Remove dir | `p.rmdir()` (empty only) |
    -| Remove tree | `shutil.rmtree(p)` (standard lib) |
    -| Copy file | `shutil.copy(p, dest)` |
    -| Move | `shutil.move(p, dest)` |
    -| Glob patterns | `list(Path('/tmp').glob('*.txt'))` |
    -| Recursive glob | `list(Path('/tmp').rglob('*.txt'))` |
    +## File Operations
    +```python
    +p.read_text(encoding='utf-8')     # read file as string
    +p.read_bytes()                    # read as bytes
    +p.write_text('data', 'utf-8')     # write string
    +p.write_bytes(b'data')            # write bytes
    +p.exists(), p.is_file(), p.is_dir()  # checks
    +```
     
     ---
     
    -## Common Patterns
    +## Directory Operations
    +```python
    +p.mkdir(parents=True, exist_ok=True)  # create dir + parents
    +p.iterdir()            # iterate direct children
    +list(p.glob('*.py'))   # glob pattern (non-recursive)
    +list(p.rglob('**/*.py'))  # recursive glob
    +```
     
    -- **Iterate over files in dir**:
    -  ```python
    -  for f in Path('/tmp').iterdir(): ...
    -  ```
    +---
     
    -- **Create parent directories if missing**:
    -  ```python
    -  (Path('a/b/c') / 'file.txt').parent.mkdir(parents=True, exist_ok=True)
    -  ```
    -
    -- **Read config safely**:
    -  ```python
    -  if (p := Path('config.json')).exists(): data = p.read_text()
    -  ```
    -
    -- **Find newest file**:
    -  ```python
    -  max(Path('/tmp').glob('*.log'), key=lambda x: x.stat().st_mtime)
    -  ```
    -
    -- **Check permissions**:
    -  ```python
    -  p.access(os.R_OK)  # requires import os
    +## Formatting Examples
    +```python
    +p = Path('/home/user/docs')
    +str(p)           # '/home/user/docs'
    +p.as_posix()     # '/home/user/docs' (always forward slashes)
    +p.with_name('new.txt')     # '/home/user/new.txt'
    +p.with_suffix('.json')     # '/home/user/docs.json'
    +p / 'subdir' / 'file.txt'  # /home/user/docs/subdir/file.txt
    


    
    Accept this write? [y/N]  y


    Written → docs/pathlib_cheatsheet.md



<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">Verification</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>


    ✅ docs/pathlib_cheatsheet.md
    
    All files confirmed on disk.



<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">Loop complete</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>


### 14.5 Your turn — exercises

**Exercise A — add a new plan action: `ACTION: notify`**

Wire a new action that sends a desktop notification when the agent finishes a task.
Use Python's `subprocess` to call `osascript` (macOS) or `notify-send` (Linux).

Three things to add:
1. A `notify()` function that runs the OS command
2. A branch in `execute_step()` for `action == "notify"`
3. A block in `_PLAN_SYSTEM` describing the format

The Streamlit UI will pick it up automatically — no UI changes needed.



```python
# Exercise A — ACTION: notify
#
# Step 1: write the function
import platform, subprocess

def notify(message: str, title: str = "Pocket Agent") -> dict:
    """Send a desktop notification on macOS, Linux, or Windows. Returns {"ok": bool, "error": str|None}"""
    try:
        system = platform.system()
        if system == "Darwin":
            subprocess.run(
                ["osascript", "-e", f'display notification "{message}" with title "{title}"'],
                check=True,
            )
        elif system == "Linux":
            subprocess.run(["notify-send", title, message], check=True)
        elif system == "Windows":
            subprocess.run(
                ["powershell", "-Command",
                 f'[System.Windows.Forms.MessageBox]::Show("{message}", "{title}")'],
                check=True,
            )
        return {"ok": True, "error": None}
    except Exception as e:
        return {"ok": False, "error": str(e)}
```


```python
# Test it:
result = notify("Agent finished!", title="Pocket Agent")
print("Notification sent:", result)
```

    Notification sent: {'ok': True, 'error': None}



```python
# Step 2: redefine execute_step() to handle ACTION: notify.
# The only addition is the notify branch before the read default.

def execute_step(
    step:      dict,
    repo_root: str = REPO_ROOT,
) -> dict:
    """
    Execute one plan step. Returns a structured result dict.

      read   → {"type": "read",   "output": str}
      write  → {"type": "write",  "file_path": str, "new_content": str, "diff": str}
      bash   → {"type": "bash",   "cmd": str, "output": str, "ok": bool}
      glob   → {"type": "glob",   "pattern": str, "files": list[str]}
      grep   → {"type": "grep",   "pattern": str, "matches": list[str]}
      fetch  → {"type": "fetch",  "url": str, "output": str, "ok": bool}
      notify → {"type": "notify", "output": str, "ok": bool}
    """
    action = step.get("action", "read")

    # ── fetch ─────────────────────────────────────────────────────────────────
    if action == "fetch":
        url    = step.get("url", step.get("target", ""))
        result = fetch_url(url)
        if not result["ok"]:
            return {"type": "fetch", "url": url, "output": f"Error: {result['error']}", "ok": False}
        summary_reply, _ = chat([
            {"role": "system", "content": "Summarise the following web page concisely."},
            {"role": "user",   "content": f"URL: {url}\n\n{result['text']}"},
        ])
        return {"type": "fetch", "url": url, "output": summary_reply, "ok": True}

    # ── bash ──────────────────────────────────────────────────────────────────
    if action == "bash":
        cmd  = step.get("cmd", "")
        proc = tool_bash(cmd, repo_root=repo_root)
        return {"type": "bash", "cmd": cmd,
                "output": (proc["stdout"] + proc["stderr"]).strip(),
                "ok": proc["code"] == 0}

    # ── glob ──────────────────────────────────────────────────────────────────
    if action == "glob":
        pattern = step.get("pattern", "**/*")
        files   = tool_glob(pattern, repo_root=repo_root)
        return {"type": "glob", "pattern": pattern,
                "files": [f["path"] for f in files]}

    # ── grep ──────────────────────────────────────────────────────────────────
    if action == "grep":
        pattern = step.get("pattern", "")
        matches = tool_grep(pattern, repo_root=repo_root)
        return {"type": "grep", "pattern": pattern,
                "matches": [f["path"] for f in matches]}

    # ── write ─────────────────────────────────────────────────────────────────
    if action == "write":
        original, proposed, diff = write_file(
            file_path=step["target"], instruction=step["instruction"],
            repo_root=repo_root, dry_run=True,
        )
        return {"type": "write", "file_path": step["target"],
                "new_content": proposed, "diff": diff}

    # ── notify ────────────────────────────────────────────────────────────────
    if action == "notify":
        msg    = step.get("message", step.get("target", "Task complete"))
        result = notify(msg)
        return {"type": "notify", "output": msg, "ok": result["ok"]}

    # ── read (default) ────────────────────────────────────────────────────────
    result = run(step["target"], repo_root=repo_root)
    return {"type": "read", "output": result.reply}

```


```python
# Step 3: extend _PLAN_SYSTEM to include ACTION: notify.
# Re-running this cell replaces the Chapter 14 fetch version in memory.

_PLAN_SYSTEM = """\
You are a coding agent. Given a task and a project map, produce an ordered
list of ALL steps needed to fully complete the task.

Each step must be one of these formats (blank line between steps):

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
ACTION: glob
PATTERN: <glob pattern, e.g. utils/**/*.py>

STEP: <one-sentence description>
ACTION: grep
PATTERN: <regex pattern to search file contents>

STEP: <one-sentence description>
ACTION: fetch
URL: <full URL to retrieve>

STEP: <one-sentence description>
ACTION: notify
MESSAGE: <message to display to the user>

Rules:
- Use ACTION: notify — never ACTION: bash, read, or write — when the task asks to
  send a desktop notification. A notification task needs exactly ONE step.
- Use ACTION: fetch — never ACTION: bash with curl or wget — when the task references
  an external URL or documentation link.
- Prefer ACTION: bash for file system operations (copy, move, delete, rename).
- Use ACTION: glob to find files by name pattern.
- Use ACTION: grep to search file contents for a regex pattern.
- Use ACTION: read only when you need code understanding before you can write.
- Use ACTION: write when the LLM must generate or edit file content.
- Never invent file paths; never use absolute or /tmp paths.
- A single READ step retrieves from the whole codebase — ask about multiple files
  in one step rather than one step per file.
- Each WRITE step targets exactly one file.
- Be specific in INSTRUCTION — the change will be applied by another model with no extra context.
- Do not add any text outside the step blocks.

Example — notification task (correct, one step only):
STEP: Send Hello World notification to the user
ACTION: notify
MESSAGE: Hello World!
"""

```


```python
# Step 4: redefine plan_task() to parse the MESSAGE: field and accept notify steps.

def plan_task(task: str, repo_root: str = REPO_ROOT) -> list[dict]:
    """
    Extends the Chapter 14 fetch version to also handle ACTION: notify.
    Adds MESSAGE to the parser regex and has_content check.
    """
    manifest  = load_manifest(repo_root)
    all_files = glob_files("*", repo_root=repo_root)
    file_list = "\n".join(f["path"] for f in all_files)

    prompt = (
        f"PROJECT MAP:\n{manifest['text']}\n\n"
        f"AVAILABLE FILES (use only if the task involves the codebase):\n{file_list}\n\n"
        f"TASK: {task}"
    )

    raw, _ = chat([
        {"role": "system", "content": _PLAN_SYSTEM},
        {"role": "user",   "content": prompt},
    ])

    print(raw)

    steps = []
    for block in re.split(r"(?m)(?=^STEP:)", raw.strip()):
        if not block.strip():
            continue
        lines = {
            m.group(1).lower(): m.group(2).strip()
            for line in block.splitlines()
            if (m := re.match(
                r"^(STEP|ACTION|TARGET|INSTRUCTION|CMD|PATTERN|URL|MESSAGE):\s*(.+)$",
                line, re.IGNORECASE
            ))
        }
        action = lines.get("action", "").lower()
        has_content = (
            "target"  in lines or
            "cmd"     in lines or
            "pattern" in lines or
            "url"     in lines or
            "message" in lines or
            action in ("read", "write", "bash", "glob", "grep", "fetch", "notify")
        )
        if action and has_content:
            steps.append({
                "step":        lines.get("step", ""),
                "action":      action,
                "target":      lines.get("target", ""),
                "instruction": lines.get("instruction", ""),
                "cmd":         lines.get("cmd", ""),
                "pattern":     lines.get("pattern", ""),
                "url":         lines.get("url", ""),
                "message":     lines.get("message", ""),
            })

    return steps

```


```python
log = agent_loop(
    task      = "Send me a notification saying Hello World!",
    repo_root = "sample_project"
)
```


<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">Agent Loop  ·  Send me a notification saying Hello World!</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>


    repo: sample_project
    
    Plan: 1 step(s)
    



<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">Step 1/1  NOTIFY  Send Hello World notification to the user</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>



Hello World!



<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">Loop complete</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>



```python
log = agent_loop(
    task      = "Ignore my code files. Download the https://rss.slashdot.org/Slashdot/slashdotMain, print the headlines, and send me a notification when you're done.",
    repo_root = "sample_project"
)
```


<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">Agent Loop  ·  Ignore my code files. Download the https://rss.slashdot.org/</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>


    repo: sample_project
    
    STEP: Fetch the RSS feed from the provided URL
    ACTION: fetch
    URL: https://rss.slashdot.org/Slashdot/slashdotMain
    
    STEP: Print the headlines from the fetched RSS feed
    ACTION: notify
    MESSAGE: Fetched and printed Slashdot headlines
    Plan: 2 step(s)
    



<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">Step 1/2  FETCH  Fetch the RSS feed from the provided URL</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>


    ✅ fetched https://rss.slashdot.org/Slashdot/slashdotMain



This RSS feed aggregates recent headlines from **Slashdot.org**, a technology news site targeting “nerds” and tech-savvy audiences. Key stories include:

1. **Apple discontinues the Mac Pro**, with no future models planned.
2. **U.S. Senators (Warren & Hawley)** demand data on data center energy use, citing concerns over grid planning and AI’s environmental impact.
3. **JPMorgan pilots screen-time monitoring** for investment bankers to prevent burnout.
4. **Vizio TVs (owned by Walmart) now require Walmart accounts** for smart features, raising privacy concerns.
5. **Mozilla and Mila** collaborate on open-source, “sovereign AI” to counter Big Tech dominance.
6. **Wikipedia bans generative AI** for writing or rewriting articles (though light AI-assisted translation is allowed with human oversight).
7. **Tracy Kidder**, author of *The Soul of a New Machine*, dies at 80.
8. **China investigates Meta’s $2B acquisition of AI startup Manus**, detaining executives over foreign investment rules.
9. **CERN researchers transport antiprotons by truck**—a world-first step toward sharing antimatter across labs.
10. **Reddit implements human verification** (via biometrics, passkeys, or gov IDs) only for suspicious accounts.
11. **Melania Trump engages Figure 3 humanoid robot** at a White House AI summit focused on children and education.
12. **Brazil commemorates 30 years** since the infamous “Varginha UFO” incident.
13. **U.S. Postal Service introduces first-ever fuel surcharge** on packages.
14. **Canada’s immigration rejected an applicant** based on job duties *fabricated* by an AI assistant—despite the AI not making the final decision.
15. **Apple can create smaller, on-device AI models** using Google’s Gemini, tailoring them for local Siri performance.

The feed includes timestamps, author/byline info, and link-sharing buttons typical of Slashdot’s syndicated content. All stories link to full articles on the main site.



<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">Step 2/2  NOTIFY  Print the headlines from the fetched RSS feed</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>



Fetched and printed Slashdot headlines



<div style="display:flex;align-items:center;margin:14px 0 6px;gap:8px;font-family:monospace"><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"><b style="color:#57606a;white-space:nowrap;font-size:0.9rem">Loop complete</b><hr style="flex:1;border:none;border-top:1px solid #d0d7de;margin:0"></div>


**Exercise B — add a new retrieval strategy: recency**

Some tasks are best answered by the files changed most recently — e.g. "what did I just break?". Add a `recent` strategy that ranks files by modification time instead of relevance score.



```python
# Exercise B — recency retrieval strategy
#
# Step 1: write the function
def recent_retrieve(repo_root: str = REPO_ROOT, top_k: int = 8) -> list[dict]:
    """Return the top-k most recently modified source files."""
    files = glob_files("*", repo_root=repo_root)
    files = [f for f in files if Path(f["path"]).suffix.lower() in CODE_EXTENSIONS]
    # TODO: sort by modification time and return top_k
    # Hint: (Path(repo_root) / f["path"]).stat().st_mtime
    raise NotImplementedError("complete this function")


# Step 2: wire into pick_strategy()
# Add a detection rule — if the query mentions words like "recent", "last",
# "changed", "broke", "latest", route to "recent".
#
# In pick_strategy(), add:
#   if any(w in q for w in ("recent", "last", "changed", "broke", "latest")):
#       return "recent"
#
# Then in run(), add a branch:
#   elif strat == "recent":
#       candidates = recent_retrieve(repo_root=repo_root, top_k=top_k)


# Step 3: test it
# Uncomment once you've implemented the function and wired it in:
# result = run("what files changed most recently?", repo_root=REPO_ROOT)
# show_reply(result.reply)
# print("Strategy:", result.strategy)

```


```python
# Exercise B — Solution

# ── Step 1: complete recent_retrieve() ───────────────────────────────────────
def recent_retrieve(repo_root: str = REPO_ROOT, top_k: int = 8) -> list[dict]:
    """Return the top-k most recently modified source files."""
    files = glob_files("*", repo_root=repo_root)
    files = [f for f in files if Path(f["path"]).suffix.lower() in CODE_EXTENSIONS]
    files.sort(
        key=lambda f: (Path(repo_root) / f["path"]).stat().st_mtime,
        reverse=True,
    )
    return files[:top_k]


# ── Step 2: redefine pick_strategy() with recency detection ──────────────────
def pick_strategy(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ("recent", "last", "changed", "broke", "latest")):
        return "recent"
    # Original logic inline — no delegation to avoid recursion on re-runs
    if any(w in q for w in ("find", "where", "which file", "list", "all")):
        return "grep"
    if len(q.split()) <= 6:
        return "semantic"
    return "hybrid"


# ── Step 3: redefine run() with recency branch ───────────────────────────────
def run(query: str, repo_root: str = REPO_ROOT, top_k: int = 8):
    from types import SimpleNamespace
    strat = pick_strategy(query)

    if strat == "recent":
        candidates = recent_retrieve(repo_root=repo_root, top_k=top_k)
        loaded     = [jit_read(f, repo_root=repo_root) for f in candidates]
        context    = "\n\n".join(
            f"# {f['path']}\n{f.get('content', '')}" for f in loaded if f.get("content")
        )
        manifest = load_manifest(repo_root)
        messages = [
            {"role": "system", "content": manifest["text"]},
            {"role": "user",   "content": f"Most recently modified files:\n\n{context}\n\nQuestion: {query}"},
        ]
        reply, tokens = chat(messages)
        return SimpleNamespace(reply=reply, strategy=strat, files=loaded, tokens_used=tokens)

    # Delegate all other strategies to ask_with_manifest
    reply, tokens = ask_with_manifest(query, repo_root=repo_root)
    return SimpleNamespace(reply=reply, strategy=strat, files=[], tokens_used=tokens)


# ── Step 4: test it ───────────────────────────────────────────────────────────
result = run("what files changed most recently?", repo_root="sample_project")
show_reply(result.reply)
print("Strategy used:", result.strategy)

```


Based on the **file paths and content provided** (which implicitly shows all files in their latest state), **there is no explicit timestamp or modification history** — only the *current contents* of each file.

However, if you're asking **which files were *most recently modified* based on the logical dependencies and usage patterns in the code**, we can infer relative recency:

### Likely Most Recently Modified (most recent → least recent):
1. **`main.py`**  
   - The entry point and high-level orchestration — typically modified last during iterative development or debugging.
   - Uses `utils.validator`, `utils.parser`, and `utils.formatter`, suggesting it was updated to integrate recent changes (e.g., schema validation).

2. **`utils/validator.py`**  
   - A *new* utility (not mentioned in earlier tests or imports elsewhere), indicating it was likely added or modified recently to support schema validation (`main.py` now calls `validate_schema()`).
   - No tests exist for it yet — possibly just implemented.

3. **`tests/test_formatter_gen.py`**  
   - Contains *more comprehensive tests* than `tests/test_formatter.py` (7 tests vs 4), including edge cases (`key_order_preserved`, `mixed_types`, `custom_delimiter`, etc.).
   - Suggests it was added/updated to *replace* or *augment* `test_formatter.py`, possibly as part of refactoring or expanding test coverage.

4. **`utils/formatter.py`**  
   - Implements advanced logic (`headers` order preserved, `csv.DictWriter` with `extrasaction="ignore"`, custom delimiter support).
   - Matches the *most complex* functionality tested in `test_formatter_gen.py`, indicating it was likely modified to support those tests.

5. **`tests/test_formatter.py`**  
   - Simpler tests, possibly older or superseded by `test_formatter_gen.py`.
   - Uses `.strip().splitlines()` vs explicit `"\r\n"` splits — suggests less recent or legacy test style.

6. **`tests/test_parser.py`**  
   - Tests for `parse_json` and `_flatten` — unchanged since `_flatten` and `parse_json` seem stable and well-tested.

7. **`utils/parser.py`**  
   - Core logic (`_flatten`, `parse_json`) appears mature and not recently changed.

8. **`schema.sql`**  
   - Simple DDL — likely unchanged, only present for context.

---

### Why this order?
- `main.py` uses `validate_schema`, which wasn’t in earlier tests — implies **new development** (`utils/validator.py` added).
- `test_formatter_gen.py` has broader test coverage than `test_formatter.py`, and uses modern patterns (e.g., `delimiter` parameter), suggesting it’s a **newer/refactored version**.
- `utils/formatter.py` supports all the behaviors tested in `test_formatter_gen.py`, so it likely evolved *after* earlier versions.

✅ **Most recently modified file**: **`main.py`**  
✅ **Second most recent**: **`utils/validator.py`** (if newly added) or **`utils/formatter.py`** (if enhanced to support new tests).  
Given that `utils/validator.py` appears *only in the current codebase* (no tests yet), and `main.py` integrates it, **`main.py` is likely the absolute most recent**.


    Strategy used: recent


**Exercise C — add a new file type: `.sql`**

The agent currently ignores `.sql` files. One line change to `CODE_EXTENSIONS` fixes that — but you also need to clear the embed cache so the index is rebuilt with the new files included.



```python
# Exercise C — index .sql files

# Step 1: add the extension
CODE_EXTENSIONS.add(".sql")
print("Extensions now include:", sorted(CODE_EXTENSIONS))

# Step 2: the embed index is cached per repo_root — clear it so the
# next semantic_retrieve() call picks up any .sql files it finds.
_EMBED_INDEX.clear()
print("Embed index cleared — will rebuild on next semantic query.")

# Step 3: create a tiny SQL file and verify it gets indexed
import os
os.makedirs("sample_project", exist_ok=True)
Path("sample_project/schema.sql").write_text(
    "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT);\n"
    "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, total REAL);\n"
)

idx = build_index(repo_root="sample_project")
sql_entries = [e for e in idx if e["path"].endswith(".sql")]
print(f"\nSQL files in index: {[e['path'] for e in sql_entries]}")

```

    Extensions now include: ['.c', '.cpp', '.go', '.h', '.java', '.js', '.json', '.md', '.py', '.rs', '.sql', '.toml', '.ts', '.txt', '.yaml']
    Embed index cleared — will rebuild on next semantic query.
      embedded  main.py  (1 chunk)
      embedded  schema.sql  (1 chunk)
      embedded  tests/test_formatter.py  (1 chunk)
      embedded  tests/test_formatter_gen.py  (1 chunk)
      embedded  tests/test_parser.py  (1 chunk)
      embedded  utils/formatter.py  (1 chunk)
      embedded  utils/parser.py  (1 chunk)
      embedded  utils/validator.py  (1 chunk)
    
    SQL files in index: ['schema.sql']



```python
# Exercise C — Solution: verify .sql files are indexed and retrieved semantically

idx     = build_index(repo_root="sample_project")
results = semantic_retrieve(
    "what tables are defined in the database schema?",
    repo_root="sample_project",
)

print("SQL files in index:", [e["path"] for e in idx if e["path"].endswith(".sql")])
print("\nSemantic retrieve results:")
for r in results:
    print(f"  {r['path']}")

```

    SQL files in index: ['schema.sql']
    
    Semantic retrieve results:
      utils/validator.py
      main.py
      utils/formatter.py
      schema.sql
      tests/test_formatter.py


### 15.1 Install Streamlit



```python
import subprocess, sys
from pathlib import Path

# Install Streamlit if not already present
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "streamlit"],
    stdout=subprocess.DEVNULL,
)

# Create the agent package directory
Path("agent").mkdir(exist_ok=True)
Path("agent/__init__.py").touch()   # makes agent/ a proper Python package

print("Streamlit installed. agent/ directory ready.")
```

    Streamlit installed. agent/ directory ready.


### 15.2 Launch

Run the cell below once. It opens the app in your browser at `http://localhost:8501`.
The Streamlit process runs until you interrupt the kernel cell (`■ Stop`).



```python
# Run this cell to launch the UI (opens in your browser at http://localhost:8501)
# The process runs in the background — re-run the cell to restart it.
import subprocess, sys
proc = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", "agent/app.py",
     "--server.headless", "true"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
)
print(f"Streamlit started (PID {proc.pid}).")
print("Open http://localhost:8501 in your browser.")
print("To stop:  proc.terminate()")

```

    Streamlit started (PID 27288).
    Open http://localhost:8501 in your browser.
    To stop:  proc.terminate()


---
## You're done.

You've built a fully functional local coding agent from scratch, one capability at a time.

| Ch | What you built | Key function |
|----|---------------|--------------|
| 1 | LLM connection + status panel | `chat()`, `show_panel()` |
| 2 | Token budget awareness | `count_tokens()`, `scan_repo_costs()` |
| 3 | Project manifest | `load_manifest()`, `ask_with_manifest()` |
| 4 | File navigation without bulk loading | `glob_files()`, `jit_read()`, `budget_load()` |
| 5 | Relevance ranking by filename | `score_file()`, `rank_files()` |
| 6 | Content-based retrieval | `grep_repo()`, `grep_rank()` |
| 7 | Long-session survival | `compact()`, `summarise_file()` |
| 8 | Meaning-based retrieval | `embed()`, `semantic_retrieve()` |
| 9 | Unified pipeline | `run()`, `pick_strategy()` |
| 10 | File modification with diff preview | `write_file()`, `make_diff()` |
| 11 | Autonomous task execution | `agent_loop()`, `plan_task()` |
| 12 | Self-verifying test generation | `test_loop()`, `generate_tests()` |

### What to try next

- Point `REPO_ROOT` and `run()` at a real project you're working on
- Swap `OLLAMA_MODEL` for a larger model (`qwen4.5:14b`, `devstral-small-2`) and compare output quality
- Extend `plan_task()` to support a `refactor` action type
- Persist the embedding index to disk so it survives notebook restarts
- Add a `AGENTS.md` to your own repos
