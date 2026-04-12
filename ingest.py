import os
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

COLLECTION_NAME = "music_knowledge"
DATA_DIR = "data"
CHUNK_SIZE = 300  # characters per chunk
CHUNK_OVERLAP = 50


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end].strip())
        start += size - overlap
    return [c for c in chunks if len(c) > 30]  # drop tiny chunks


def load_documents(data_dir: str) -> list[dict]:
    """Load all .txt and .pdf files from data directory."""
    docs = []
    for path in Path(data_dir).glob("**/*"):
        if path.suffix == ".txt":
            text = path.read_text(encoding="utf-8")
            docs.append({"text": text, "source": path.name})
        elif path.suffix == ".pdf":
            import PyPDF2
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = "\n".join(p.extract_text() for p in reader.pages)
            docs.append({"text": text, "source": path.name})
    return docs


def ingest():
    print("Loading embedding model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")  # fast, good quality, free

    print("Setting up ChromaDB...")
    client = chromadb.PersistentClient(path="./chroma_db")

    # Wipe and recreate for a clean ingest
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    print(f"Loading documents from {DATA_DIR}/...")
    docs = load_documents(DATA_DIR)
    if not docs:
        print("No documents found. Add .txt or .pdf files to the data/ folder.")
        return

    all_chunks, all_ids, all_metadata = [], [], []
    chunk_id = 0

    for doc in docs:
        chunks = chunk_text(doc["text"])
        print(f"  {doc['source']}: {len(chunks)} chunks")
        for chunk in chunks:
            all_chunks.append(chunk)
            all_ids.append(f"chunk_{chunk_id}")
            all_metadata.append({"source": doc["source"]})
            chunk_id += 1

    print(f"Embedding {len(all_chunks)} chunks...")
    embeddings = embedder.encode(all_chunks, show_progress_bar=True).tolist()

    print("Storing in ChromaDB...")
    collection.add(
        documents=all_chunks,
        embeddings=embeddings,
        ids=all_ids,
        metadatas=all_metadata,
    )
    print(f"Done. {len(all_chunks)} chunks stored.")


if __name__ == "__main__":
    ingest()
