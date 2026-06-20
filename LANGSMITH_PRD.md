# PRD: LangSmith Monitoring & Tracing

## Context

The RAG pipeline (`ingest → retriever → rag`) currently has **no observability**. When a recommendation is poor, there's no way to tell whether the retriever returned bad chunks, the prompt was malformed, or the LLM ignored its context — the only signal is the final printed answer and a basic pass/fail in `test_rag.py`.

LangSmith captures a structured **trace** for every query: the query in, the chunks retrieved (with scores), the assembled prompt, the LLM output, token usage, and latency at each step. These traces also feed LangSmith's monitoring dashboards (latency / volume / error trends over time).

**Key constraint:** this project uses `ollama` + `sentence-transformers` + `chromadb` **directly — there is no LangChain**, so LangSmith's automatic LangChain integration does not apply. We instrument manually with the `langsmith` SDK's `@traceable` decorator, which is a **no-op when tracing is disabled** (the app still runs offline without a key).

**Decisions (confirmed with user):**
- LLM call: keep the existing `ollama` package, wrap it in a manual `@traceable(run_type="llm")` function that logs token counts from ollama's response.
- Scope: full query pipeline **plus** the `test_rag.py` eval harness.

## How tracing works here (reference)

- Enabled by env vars: `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, optional `LANGSMITH_PROJECT`. When set, `@traceable` functions auto-send traces; when unset they run as plain functions.
- `@traceable(run_type=...)` creates a span nested under whatever traced function is currently running. Decorating `ask()` as the root makes `retrieve`, `build_context`, and the LLM call appear as child spans automatically.
- `run_type` values used: `"chain"` (root), `"retriever"`, `"prompt"`, `"llm"`.

## Requirements

### R1 — Dependencies (`requirements.txt`)
Add `langsmith` and `python-dotenv`.

### R2 — Secrets (`.env.example` committed, `.env` local/gitignored)
`.gitignore` already excludes `.env`. Create `.env.example`:
```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-key-here
LANGSMITH_PROJECT=music-rec-rag
```
Load it once at the top of `rag.py` (imported by both the CLI and `test_rag.py`) via `load_dotenv()`.

### R3 — Retriever span (`retriever.py`)
Decorate `Retriever.retrieve` with `@traceable(run_type="retriever")` so the query and returned chunks (with scores) are captured.

### R4 — Pipeline spans (`rag.py`)
- Extract the `ollama.chat(...)` call into a `generate(messages)` function decorated `@traceable(run_type="llm")`; attach token usage from ollama's `prompt_eval_count` / `eval_count` to the span via `get_current_run_tree()` so usage/cost panels populate.
- Decorate `build_context` with `@traceable(run_type="prompt")`.
- Decorate `ask` with `@traceable(run_type="chain")` (root); call `generate(...)` instead of `ollama.chat(...)` inline. Attach model + top_k as metadata.
- Resulting trace tree per query: `ask → retrieve → build_context → generate`.

### R5 — Eval harness (`test_rag.py`)
Tag each test-case run via `langsmith_extra={"tags": ["eval"], "metadata": {"test_case": ...}}` so eval batches are grouped and filterable separately from production traffic.

## Files touched
- `requirements.txt` — add `langsmith`, `python-dotenv`
- `.env.example` — **new**, committed placeholders
- `retriever.py` — `@traceable(run_type="retriever")` on `retrieve`
- `rag.py` — `load_dotenv()`, new `generate()` LLM span, decorate `build_context` + `ask`
- `test_rag.py` — tag eval runs via `langsmith_extra`

## Verification
1. `pip install -r requirements.txt`
2. `cp .env.example .env` and paste the real `LANGSMITH_API_KEY`.
3. Ensure data is ingested (`python ingest.py`) and `ollama serve` is running.
4. `python rag.py`, ask *"recommend something melancholic and acoustic"*. In LangSmith → project **music-rec-rag**, confirm one trace with nested spans `ask → retrieve → build_context → generate`, the retriever span showing chunks/scores, and the LLM span showing token counts.
5. `python test_rag.py`; confirm a batch of traces tagged `eval` appears, filterable by the `eval` tag.
6. **Graceful degradation:** unset `LANGSMITH_TRACING` (or remove the key) and rerun `python rag.py` — the app must work normally with no traces sent and no errors.

## Out of scope (follow-on)
- LangSmith Datasets + `evaluate()` for scored regression runs (pairs with the RAGAS item in `IMPROVEMENTS.md`).
- Native document rendering for the retriever span (reshape outputs to `{page_content, metadata}`).
