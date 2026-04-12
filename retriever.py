import chromadb
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "music_knowledge"


class Retriever:
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_collection(COLLECTION_NAME)

    def retrieve(self, query: str) -> list[dict]:
        """Embed the query and return top-k most similar chunks."""
        query_embedding = self.embedder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "text": doc,
                "source": meta["source"],
                "score": round(1 - dist, 3),  # convert distance to similarity
            })
        return chunks
