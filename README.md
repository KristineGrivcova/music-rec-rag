# Music Recommendation RAG

A local retrieval-augmented generation (RAG) system that answers natural language questions about music and recommends albums from a curated knowledge base. Runs entirely on your machine — no API keys or internet connection required after setup.

## How it works

```
Your question
     │
     ▼
Embed query (sentence-transformers)
     │
     ▼
Vector search → top-4 matching chunks (ChromaDB)
     │
     ▼
Inject chunks into prompt
     │
     ▼
Local LLM generates grounded answer (Ollama / Mistral)
     │
     ▼
Response + sources shown
```

## Project structure

```
music-rec-rag/
├── data/
│   └── albums.txt      # Knowledge base — add your own albums here
├── ingest.py           # Loads, chunks, embeds, and stores documents
├── retriever.py        # Semantic search over the vector store
├── rag.py              # CLI chatbot — the main entry point
├── test_rag.py         # Retrieval and generation evaluation
└── requirements.txt
```

## Setup

### 1. Create and activate the conda environment

```bash
conda create -n music-rag python=3.11 -y
conda activate music-rag
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Ollama and pull a model

```bash
brew install ollama      # macOS
ollama pull mistral      # ~4GB download
```

### 4. Ingest the knowledge base

```bash
python ingest.py
```

This chunks the albums in `data/albums.txt`, embeds them with `all-MiniLM-L6-v2`, and stores them in a local ChromaDB database (`chroma_db/`).

### 5. Start the chatbot

```bash
# In one terminal — start the local model server
ollama serve

# In another terminal
python rag.py
```

## Example queries

```
You: recommend something melancholic and acoustic
You: what's similar to Radiohead but warmer?
You: something good for a rainy day
You: suggest an album with electronic textures and guitar
```

## Changing the model

Edit the `MODEL` variable at the top of `rag.py`:

```python
MODEL = "mistral"  # default
```

| Model | Size | Notes |
|---|---|---|
| `mistral` | ~4 GB | Good balance of quality and speed |
| `llama3.2` | ~2 GB | Smaller and faster |
| `phi4` | ~9 GB | Higher quality, slower |
| `gemma3:1b` | ~1 GB | Tiny, very fast |

Pull any model with `ollama pull <model-name>` before switching.

## Adding more albums

Append entries to `data/albums.txt` in the same format:

```
Album Title - Artist (Year)
Genre: ...
Mood: ...
Instrumentation: ...
Key tracks: ...
Description: ...
```

Then re-run `python ingest.py` to rebuild the vector store.

## Running the tests

```bash
python test_rag.py
```

Runs retrieval checks (relevance score threshold) and generation checks (answer grounding and keyword presence) against three test queries.

## Stack

| Component | Library |
|---|---|
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Vector store | `ChromaDB` (persisted locally) |
| LLM | `Ollama` — Mistral (or any pulled model) |
| CLI interface | `rich` |
