"""RAG-based summarizer agent.

This module provides helpers to generate a concise summary (<= 200 words)
for a document using retrieval-augmented generation (RAG). The summarizer:
- Loads and chunks documents using `rag.loader.load_and_split` (RecursiveCharacterTextSplitter).
- Indexes chunks into an in-memory FAISS store (via `rag.vectorstore.FaissVectorStore`).
- Retrieves the most relevant chunks for the summarization task and sends
  them to an LLM with a tightly-constrained prompt that instructs the model
  to only use the provided passages and to avoid inventing facts.

Avoiding hallucinations (prompt design):
- The LLM is explicitly told to use ONLY the provided passages. This reduces
  reliance on model world knowledge and limits generation to source material.
- The prompt demands that if the provided passages don't contain enough
  information, the model must reply with a short, explicit statement like
  "Insufficient information to produce a reliable summary." rather than
  inventing details.
- Temperature is set to 0 (deterministic) where possible.
"""

from __future__ import annotations

from typing import List, Sequence, Optional, Callable, Dict, Any
from pathlib import Path
import logging

from rag.loader import load_and_split
from rag.vectorstore import FaissVectorStore

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from agents.llm import get_ollama_predict
# We enforce local-only LLM usage; do not fallback to cloud LLMs.


def _build_rag_prompt(passages: Sequence[str], max_words: int = 200) -> str:
    """Construct the RAG prompt that instructs the LLM to summarize using only passages.

    Prompt logic and choices:
    - We enumerate passages so the model can reference small, numbered excerpts.
    - We explicitly forbid using outside knowledge and require the model to
      state "Insufficient information..." if the passages lack key facts.
    - We constrain the summary length by word count and ask for a plain-text
      summary with no citations or added text.
    """
    passages_block = "\n\n".join(f"PASSAGE {i+1}:\n{p.strip()}" for i, p in enumerate(passages))

    prompt = (
        "You are a careful summarization assistant.\n"
        "Use ONLY the text in the provided PASSAGE blocks below to produce a concise summary.\n"
        "Do NOT use any external knowledge, world facts, or assumptions beyond these passages.\n"
        "If the passages do not provide enough information to create a reliable summary, reply exactly:\n"
        "Insufficient information to produce a reliable summary.\n"
        "Otherwise, produce a single-paragraph summary of the content in at most "
        f"{max_words} words. Do NOT include citations, bullet lists, or any extra commentary.\n\n"
        "PASSAGES:\n\n"
        f"{passages_block}\n\n"
        "Provide the summary now (plain text, <= {max_words} words):"
    )
    return prompt


def summarize_text(
    text: str,
    llm_predict: Optional[Callable[[str], str]] = None,
    max_words: int = 200,
    top_k: int = 6,
    temperature: float = 0.0,
) -> str:
    """Summarize a raw text string using RAG retrieval over chunks.

    Per project rules, raw pasted text MUST be chunked with
    RecursiveCharacterTextSplitter (not using file loaders).
    """
    # Use project-provided text splitter (langchain or fallback)
    from rag.loader import get_text_splitter

    splitter = get_text_splitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(text)

    if not chunks:
        return "Insufficient information to produce a reliable summary."

    # Build a FAISS store from the chunks (use chunk text as documents)
    store = FaissVectorStore()
    metadatas = [{"text": c} for c in chunks]
    store.add_documents([m["text"] for m in metadatas], metadatas)

    # Retrieve top_k passages for the summarization query
    query = "Summarize the document concisely."
    results = store.search(query, k=top_k)

    passages: List[str] = []
    for md, score in results:
        p_text = md.get("text") or md.get("page_content") or ""
        if p_text:
            passages.append(p_text)

    if not passages:
        return "Insufficient information to produce a reliable summary."

    prompt = _build_rag_prompt(passages, max_words=max_words)

    if llm_predict is not None:
        raw = llm_predict(prompt)
    else:
        ollama = get_ollama_predict()
        if ollama is None:
            raise RuntimeError("Local Ollama model not available. Install and run Ollama and ensure qwen2.5:3b is pulled.")
        raw = ollama(prompt)

    if raw is None:
        raise RuntimeError("LLM produced no output")

    summary = raw.strip()

    # Enforce word limit as a last-resort safeguard: truncate to max_words
    words = summary.split()
    if len(words) > max_words:
        logger.debug("Summary exceeds %d words; truncating to word limit.", max_words)
        summary = " ".join(words[:max_words]).rstrip() + "..."

    return summary


def summarize_document(
    path: str,
    llm_predict: Optional[Callable[[str], str]] = None,
    max_words: int = 200,
    top_k: int = 6,
    temperature: float = 0.0,
) -> str:
    """Load a file and summarize it using RAG.

    Args:
        path: Path to a supported file (pdf, txt, docx). Uses `rag.loader`.
        llm_predict: Optional callable(prompt) used to produce the summary.
        max_words: Maximum allowed words in the summary.
        top_k: How many retrieved passages to include.
        temperature: LLM temperature.

    Returns:
        Summary string.
    """
    # Use loader to get chunks for the file path
    docs = load_and_split(path, chunk_size=800, chunk_overlap=100)

    # Prepare docs for vector store
    doc_dicts: List[Dict[str, Any]] = []
    for d in docs:
        content = getattr(d, "page_content", None) or getattr(d, "text", None) or str(d)
        md = dict(getattr(d, "metadata", {}) or {})
        md.setdefault("text", content)
        doc_dicts.append(md)

    store = FaissVectorStore()
    texts = [md["text"] for md in doc_dicts]
    store.add_documents(texts, doc_dicts)

    # Use the same RAG summarization path: retrieve and summarize
    query = "Summarize the document concisely."
    results = store.search(query, k=top_k)
    passages = [md.get("text") for md, _ in results if md.get("text")]

    if not passages:
        return "Insufficient information to produce a reliable summary."

    prompt = _build_rag_prompt(passages, max_words=max_words)

    if llm_predict is not None:
        raw = llm_predict(prompt)
    else:
        ollama = get_ollama_predict()
        if ollama is None:
            raise RuntimeError("Local Ollama model not available. Install and run Ollama and ensure qwen2.5:3b is pulled.")
        raw = ollama(prompt)

    if raw is None:
        raise RuntimeError("LLM produced no output")

    summary = raw.strip()
    words = summary.split()
    if len(words) > max_words:
        logger.debug("Summary exceeds %d words; truncating to word limit.", max_words)
        summary = " ".join(words[:max_words]).rstrip() + "..."

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize a document using RAG retrieval")
    parser.add_argument("path", help="Path to file or '-' to read from stdin")
    parser.add_argument("--max-words", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()

    if args.path == "-":
        import sys

        text = sys.stdin.read()
        print(summarize_text(text, max_words=args.max_words, top_k=args.top_k))
    else:
        print(summarize_document(args.path, max_words=args.max_words, top_k=args.top_k))
