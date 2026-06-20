from dotenv import load_dotenv

load_dotenv()  # load LangSmith env vars (LANGSMITH_TRACING/API_KEY/PROJECT) before tracing

import ollama
from langsmith import traceable, get_current_run_tree
from retriever import Retriever
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

MODEL = "mistral"  # change to "llama3.2", "phi4", etc. — must be pulled in Ollama first

console = Console()

SYSTEM_PROMPT = """You are a music recommendation assistant with access to a curated knowledge base of albums.

When answering, you must:
1. Base your answer ONLY on the provided context chunks
2. If the context doesn't contain enough information, say so honestly
3. Always explain WHY a recommendation fits the user's request
4. Be specific — reference instrumentation, mood, and era from the context

If asked for a recommendation, provide 1-3 albums with a brief explanation for each."""


@traceable(run_type="llm", metadata={"ls_provider": "ollama", "ls_model_name": MODEL})
def generate(messages: list[dict]) -> str:
    """Call the local LLM and surface token usage to LangSmith."""
    response = ollama.chat(model=MODEL, messages=messages)

    # Attach ollama's token counts so LangSmith's usage/cost panels populate.
    run = get_current_run_tree()
    if run is not None:
        run.metadata["usage_metadata"] = {
            "input_tokens": response.get("prompt_eval_count", 0),
            "output_tokens": response.get("eval_count", 0),
        }

    return response["message"]["content"]


@traceable(run_type="prompt")
def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context block for the prompt."""
    if not chunks:
        return "No relevant context found."
    lines = []
    for i, chunk in enumerate(chunks, 1):
        lines.append(f"[Source {i} — {chunk['source']} (relevance: {chunk['score']})]")
        lines.append(chunk["text"])
        lines.append("")
    return "\n".join(lines)


@traceable(run_type="chain", metadata={"model": MODEL})
def ask(query: str, retriever: Retriever, history: list[dict]) -> tuple[str, list[dict]]:
    """Single RAG turn: retrieve → build prompt → generate."""
    chunks = retriever.retrieve(query)
    context = build_context(chunks)

    user_message = f"""User question: {query}

Relevant knowledge base context:
---
{context}
---

Answer based on the context above."""

    history.append({"role": "user", "content": user_message})

    answer = generate(
        [{"role": "system", "content": SYSTEM_PROMPT}] + history,
    )

    # Store the clean query (not the context-injected version) for history
    history[-1] = {"role": "user", "content": query}
    history.append({"role": "assistant", "content": answer})

    return answer, chunks


def main():
    console.print(Panel.fit(
        f"[bold]Music RAG[/bold] — local model: [cyan]{MODEL}[/cyan] — ask me about albums or for recommendations",
        border_style="dim"
    ))

    retriever = Retriever(top_k=4)
    history = []

    while True:
        try:
            query = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\nBye.")
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit", "q"):
            break

        with console.status(f"Retrieving and generating with {MODEL}..."):
            answer, chunks = ask(query, retriever, history)

        console.print("\n[bold green]Assistant:[/bold green]")
        console.print(Markdown(answer))

        console.print(f"\n[dim]Sources used: {', '.join(set(c['source'] for c in chunks))}[/dim]")


if __name__ == "__main__":
    main()
