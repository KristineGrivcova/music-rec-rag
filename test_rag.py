"""
Basic RAG evaluation. Tests retrieval quality and answer grounding.
Run with: python test_rag.py
"""
from retriever import Retriever
from rag import ask, build_context

TEST_CASES = [
    {
        "query": "recommend something melancholic and acoustic",
        "expected_sources": ["albums.txt"],
        "must_contain": ["acoustic", "melanchol"],  # substring match
    },
    {
        "query": "what album is best for a rainy day",
        "expected_sources": ["albums.txt"],
        "must_contain": [],
    },
    {
        "query": "something with electronic textures and guitar",
        "expected_sources": ["albums.txt"],
        "must_contain": [],
    },
]


def test_retrieval(retriever: Retriever):
    print("\n── Retrieval tests ─────────────────────────")
    for tc in TEST_CASES:
        chunks = retriever.retrieve(tc["query"])
        sources = [c["source"] for c in chunks]
        scores = [c["score"] for c in chunks]
        top_score = max(scores) if scores else 0

        passed = top_score > 0.2  # minimum relevance threshold
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] '{tc['query'][:45]}...'")
        print(f"       Top score: {top_score:.3f} | Sources: {set(sources)}")


def test_generation(retriever: Retriever):
    print("\n── Generation tests ────────────────────────")
    for tc in TEST_CASES:
        history = []
        answer, chunks = ask(
            tc["query"],
            retriever,
            history,
            langsmith_extra={"tags": ["eval"], "metadata": {"test_case": tc["query"]}},
        )
        answer_lower = answer.lower()

        grounding_ok = len(chunks) > 0
        content_ok = all(kw.lower() in answer_lower for kw in tc["must_contain"])

        passed = grounding_ok and content_ok
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] '{tc['query'][:45]}...'")
        print(f"       Grounded: {grounding_ok} | Keywords: {content_ok}")
        print(f"       Answer preview: {answer[:120].strip()}...")
        print()


def main():
    retriever = Retriever(top_k=4)
    test_retrieval(retriever)
    test_generation(retriever)
    print("── Done ────────────────────────────────────")


if __name__ == "__main__":
    main()
