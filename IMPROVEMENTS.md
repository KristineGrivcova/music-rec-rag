# Improvement Plan

---

## Tier 1 — Quick wins (high impact, low effort)

### 1. Semantic chunking
**File:** `ingest.py`

**Problem:** The 300-character sliding window splits albums mid-sentence. A ~500-char entry like Nick Drake's Pink Moon is cut into 2 incoherent fragments, separating Mood from Description. The retriever then returns half-albums.

**Fix:** Split `albums.txt` on double newlines instead — each album is already a natural semantic unit. Store one ChromaDB document per album. Also extract structured metadata (artist, year, genre, mood) as ChromaDB fields so they're available for filtering later.

```python
def parse_albums(text: str) -> list[dict]:
    entries = [e.strip() for e in text.split("\n\n") if e.strip()]
    albums = []
    for entry in entries:
        lines = entry.splitlines()
        header = lines[0]  # "OK Computer - Radiohead (1997)"
        meta = {"source": "albums.txt"}
        # extract artist/year from header, genre/mood from labelled lines
        albums.append({"text": entry, "metadata": meta})
    return albums
```

---

### 2. Config file
**File:** new `config.py`

**Problem:** Constants are scattered across four files. `top_k` is 5 in `retriever.py`, 4 in `rag.py`, 4 in `test_rag.py`. The embedding model name is hardcoded in two places. The ChromaDB path `"./chroma_db"` is a relative string in two files — it breaks if you run from a different directory.

**Fix:** One config file, everything imported from it.

```python
from pathlib import Path

ROOT = Path(__file__).parent

# Paths
CHROMA_PATH = str(ROOT / "chroma_db")
DATA_DIR    = str(ROOT / "data")

# Collection
COLLECTION_NAME = "music_knowledge"

# Embedding
EMBED_MODEL = "all-MiniLM-L6-v2"

# Retrieval
RETRIEVAL_TOP_K = 10   # candidates fed to re-ranker
FINAL_TOP_K     = 4    # results passed to LLM after re-ranking

# Re-ranking
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Generation
OLLAMA_MODEL = "mistral"
MAX_TOKENS   = 1024
TEMPERATURE  = 0.7
MAX_HISTORY  = 20      # max turns kept in memory
```

---

### 3. Streaming responses
**File:** `rag.py`

**Problem:** `ollama.chat()` buffers the entire response before anything is printed. For a 200-word answer on a slow model this is 5–10 seconds of silence.

**Fix:** Use `stream=True` and print tokens as they arrive with `rich`'s `Live`.

```python
from rich.live import Live

with Live(console=console, refresh_per_second=15) as live:
    full_answer = ""
    for chunk in ollama.chat(model=MODEL, messages=messages, stream=True):
        full_answer += chunk["message"]["content"]
        live.update(Markdown(full_answer))
```

---

### 4. Ollama error handling
**File:** `rag.py`

**Problem:** If Ollama isn't running, the chatbot crashes with a raw `httpx.ConnectError` stack trace.

**Fix:** Catch the connection error and print a helpful message.

```python
import httpx

try:
    response = ollama.chat(...)
except httpx.ConnectError:
    console.print("[red]Ollama is not running. Start it with: ollama serve[/red]")
    return "", []
```

---

## Tier 2 — CV-worthy additions (moderate effort, high value)

### 5. Cross-encoder re-ranking
**Files:** new `reranker.py`, update `rag.py`

**Problem:** Vector similarity scores documents independently — query and album are embedded separately and compared by distance. The model never sees them together, so it can miss nuanced relevance (e.g. "acoustic but not too sparse" vs "sparse acoustic").

**Fix:** Retrieve top-10 candidates from ChromaDB, then re-score each against the full query using a cross-encoder. The cross-encoder takes `[query, album_text]` as a single input and outputs a fine-grained relevance score. Return the top-4 to the LLM.

```python
# reranker.py
from sentence_transformers import CrossEncoder
from config import RERANKER_MODEL, FINAL_TOP_K

class Reranker:
    def __init__(self):
        self.model = CrossEncoder(RERANKER_MODEL)

    def rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        pairs = [(query, c["text"]) for c in chunks]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        return [c for _, c in ranked[:FINAL_TOP_K]]
```

Pipeline becomes: retrieve 10 → re-rank → top 4 → LLM.

---

### 6. Metadata filtering
**Files:** `ingest.py`, `retriever.py`

**Problem:** ChromaDB supports metadata filtering but none is stored beyond the source filename. Queries like "something from the 70s" or "only folk albums" have no way to pre-filter results.

**Fix:** Store structured fields during ingest:

```python
metadata = {
    "source":   "albums.txt",
    "artist":   "Joni Mitchell",
    "year":     1971,
    "genre":    "Folk, singer-songwriter",
    "mood":     "Raw, vulnerable, bittersweet",
}
```

Then allow optional filters at retrieval time:

```python
def retrieve(self, query: str, where: dict = None) -> list[dict]:
    results = self.collection.query(
        query_embeddings=[embedding],
        n_results=self.top_k,
        where=where,  # e.g. {"year": {"$lt": 1980}}
    )
```

---

### 7. Conversation memory with sliding window
**File:** `rag.py`

**Problem:** The full conversation history is passed to Ollama on every turn. A long session will eventually exceed the model's context window, causing truncation or errors.

**Fix:** Cap history at `MAX_HISTORY` turns (from config). Optionally, summarise older turns rather than discarding them.

```python
from config import MAX_HISTORY

# After appending new messages:
if len(history) > MAX_HISTORY:
    history = history[-MAX_HISTORY:]
```

---

## Tier 3 — Advanced (high effort, senior-level territory)

### 8. Hybrid search (BM25 + vector)
**Files:** `retriever.py`, new `bm25_index.py`

**Problem:** Vector search catches vibe-level similarity but misses exact keyword matches. A query for "Paranoid Android" should find OK Computer directly, but the embedding might miss the exact track name. BM25 keyword search catches it.

**Fix:** Build a BM25 index over the album corpus alongside ChromaDB. At retrieval time, run both searches and fuse the result lists using Reciprocal Rank Fusion (RRF).

```python
# RRF fusion
def fuse(vector_results, bm25_results, k=60):
    scores = {}
    for rank, doc in enumerate(vector_results):
        scores[doc["id"]] = scores.get(doc["id"], 0) + 1 / (k + rank)
    for rank, doc in enumerate(bm25_results):
        scores[doc["id"]] = scores.get(doc["id"], 0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

Library: `rank_bm25` (`pip install rank-bm25`)

---

### 9. RAGAS evaluation
**File:** `test_rag.py`

**Problem:** The current test suite checks a relevance score threshold (`> 0.2`) and substring presence — both trivially passable. There's no real measure of whether answers are faithful to the retrieved context or actually useful.

**Fix:** Integrate RAGAS to score the pipeline on four metrics:
- **Faithfulness** — does the answer only use information from the context?
- **Answer relevance** — does the answer address the question?
- **Context precision** — are the retrieved chunks actually relevant?
- **Context recall** — did retrieval capture the right information?

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

dataset = Dataset.from_dict({
    "question": [tc["query"] for tc in TEST_CASES],
    "answer":   answers,
    "contexts": [[c["text"] for c in chunk_list] for chunk_list in all_chunks],
})
result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
print(result)
```

Library: `pip install ragas`

---

### 10. Streamlit UI
**File:** new `app.py`

**Problem:** The CLI is functional but not shareable or demonstrable to non-technical people.

**Fix:** Wrap `rag.py` in a Streamlit app (~50 lines). Gives you a chat interface in a browser, source attribution cards, and a sidebar for switching models — all without a backend server.

```python
import streamlit as st
from retriever import Retriever
from reranker import Reranker
from rag import ask, build_context

st.title("Music RAG")
retriever = Retriever()
reranker = Reranker()

if "history" not in st.session_state:
    st.session_state.history = []

query = st.chat_input("Ask about music...")
if query:
    answer, chunks = ask(query, retriever, reranker, st.session_state.history)
    st.chat_message("assistant").write(answer)
```

`pip install streamlit` then `streamlit run app.py`

---

---

### 11. HuggingFace Spaces deployment
**Files:** `rag.py`, new `app.py`

**Problem:** The current system relies on Ollama (`ollama serve`), a local process that can't run on HuggingFace Spaces containers. Spaces needs an API-based LLM backend.

**Fix:** Abstract the LLM call behind a `generate()` function in `rag.py`, controlled by a `LLM_BACKEND` environment variable. Locally it routes to Ollama unchanged; on Spaces it routes to the HuggingFace Inference API (free tier).

```python
# rag.py
import os

def generate(messages: list[dict]) -> str:
    backend = os.getenv("LLM_BACKEND", "ollama")
    if backend == "huggingface":
        from huggingface_hub import InferenceClient
        client = InferenceClient(
            model=os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2"),
            token=os.getenv("HF_TOKEN"),
        )
        return client.chat_completion(messages=messages).choices[0].message.content
    else:
        return ollama.chat(model=OLLAMA_MODEL, messages=messages)["message"]["content"]
```

**ChromaDB:** Swap `PersistentClient` → `EphemeralClient` on Spaces (no disk persistence between restarts). Re-ingest from `albums.txt` at app startup automatically.

**New files:**
- `app.py` — Streamlit chat UI, works both locally and on Spaces

**Env vars to set on HuggingFace Spaces:**
- `HF_TOKEN` — your HuggingFace API token (free tier works)
- `LLM_BACKEND=huggingface`
- `HF_MODEL=mistralai/Mistral-7B-Instruct-v0.2`

**New dependency:** add `huggingface_hub` to `requirements.txt`

---

## Summary

| # | Improvement | Files | Effort | Impact |
|---|---|---|---|---|
| 1 | Semantic chunking | `ingest.py` | Low | High |
| 2 | Config file | new `config.py` | Low | Medium |
| 3 | Streaming responses | `rag.py` | Low | Medium |
| 4 | Ollama error handling | `rag.py` | Low | Low |
| 5 | Re-ranking | new `reranker.py`, `rag.py` | Medium | High |
| 6 | Metadata filtering | `ingest.py`, `retriever.py` | Medium | Medium |
| 7 | Sliding window memory | `rag.py` | Low | Medium |
| 8 | Hybrid search (BM25) | `retriever.py`, new `bm25_index.py` | High | High |
| 9 | RAGAS evaluation | `test_rag.py` | Medium | Medium |
| 10 | Streamlit UI | new `app.py` | Medium | Medium |
| 11 | HF Spaces deployment | `rag.py`, new `app.py` | Medium | Medium |
