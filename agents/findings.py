"""Extract key findings from documents as bullet points.

This module provides helpers to extract concise, factual key findings
from text or files. It follows a retrieval-augmented approach:
- Chunk the input into overlapping passages (to preserve context)
- Index passages into an in-memory FAISS store and retrieve the most
  relevant passages for the intent "key findings"
- Ask an LLM to return only bullet points (one finding per bullet)

Prompt design notes (why/how):
- We explicitly instruct the model to return ONLY bullet points. This
  reduces verbosity and unwanted commentary.
- We demand that the model uses ONLY the provided passages and must
  respond with "No key findings found." when passages contain no
  extractable findings. This reduces hallucinations.
- We limit the number of findings via `max_findings` and ask for
  short, factual bullets (1-2 sentences each).
"""

from __future__ import annotations

from typing import List, Sequence, Optional, Callable, Tuple, Dict, Any
from pathlib import Path
import logging
import re

from rag.vectorstore import FaissVectorStore

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from agents.llm import get_ollama_predict
# Enforce local-only LLM usage; do not fallback to cloud LLMs.


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """Simple character-based chunker with overlap.

    We use a character-based splitter to avoid a hard dependency on
    langchain splitters for this helper. The chunk_size/overlap match
    the rest of the project defaults.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        if end == text_len:
            break
        start = max(0, end - overlap)

    return chunks


def _build_prompt(passages: Sequence[str], max_findings: int = 6) -> str:
    """Create the prompt asking the LLM to produce bullet-point findings.

    Key instructions:
    - Use ONLY the passages provided.
    - Return ONLY bullet points (each on its own line, prefixed with '-').
    - If no findings, respond exactly 'No key findings found.' (without quotes).
    - Return at most `max_findings` bullets.
    - Keep each bullet short and factual (1-2 sentences).
    """
    passages_block = "\n\n".join(f"PASSAGE {i+1}:\n{p.strip()}" for i, p in enumerate(passages))
    prompt = (
        "You are a precise extractor of key findings.\n"
        "Use ONLY the text in the PASSAGE blocks below. Do NOT use any outside knowledge.\n"
        "Return ONLY bullet points (one per line) describing key findings. Each bullet must be prefixed with a single hyphen and a space ('- ').\n"
        "Return at most "
        f"{max_findings} bullets. Each bullet should be 1-2 short factual sentences.\n"
        "If there are no extractable findings in the passages, reply EXACTLY: No key findings found.\n\n"
        "When processing legal documents, prioritize clauses, obligations, parties, and dates.\n"
        "When processing transactional or invoice-like documents, prioritize amounts, parties, dates, and identifiers (invoice numbers, transaction IDs).\n\n"
        "PASSAGES:\n\n"
        f"{passages_block}\n\n"
        "Now list the key findings as described above."
    )
    return prompt


def _parse_bullets(raw: str) -> List[str]:
    """Parse a raw model response into a list of cleaned bullet strings."""
    if not raw:
        return []

    text = raw.strip()
    # Exact no-findings sentinel
    if text.strip() == "No key findings found.":
        return []

    lines = text.splitlines()
    bullets: List[str] = []
    bullet_re = re.compile(r"^\s*[-*•]\s+(.*)$")
    numbered_re = re.compile(r"^\s*\d+[.)]\s+(.*)$")

    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        m = bullet_re.match(ln)
        if m:
            bullets.append(m.group(1).strip())
            continue
        m2 = numbered_re.match(ln)
        if m2:
            bullets.append(m2.group(1).strip())
            continue
        # If line doesn't start with bullet, but looks like a short sentence,
        # accept it as a fallback.
        if len(ln) < 240:
            bullets.append(ln)

    # Final cleanup: remove empty or duplicate bullets while preserving order
    seen = set()
    cleaned: List[str] = []
    for b in bullets:
        if not b:
            continue
        if b in seen:
            continue
        seen.add(b)
        cleaned.append(b)
    return cleaned


def extract_findings_from_text(
    text: str,
    llm_predict: Optional[Callable[[str], str]] = None,
    max_findings: int = 6,
    top_k: int = 8,
    temperature: float = 0.0,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> List[str]:
    """Extract key findings from a block of text and return bullet points.

    Args:
        text: Full document text.
        llm_predict: Optional callable(prompt) -> raw_text. If omitted and
            LangChain is installed, uses Ollama with the given temperature.
        max_findings: Maximum number of bullet findings to return.
        top_k: Number of retrieved passages to include in the prompt.
        temperature: LLM temperature (0 for deterministic behavior).
        chunk_size/chunk_overlap: Chunking parameters used to build RAG store.

    Returns:
        List of bullet-point strings (possibly empty if none found).
    """
    # Use project-provided text splitter (langchain or fallback)
    from rag.loader import get_text_splitter

    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    if not chunks:
        return []

    # Build FAISS store and add chunks
    store = FaissVectorStore()
    # metadata at minimum contains 'text'
    metadatas = [{"text": c} for c in chunks]
    store.add_documents([m["text"] for m in metadatas], metadatas)
    # Retrieve the most relevant passages for the 'key findings' intent
    query = "Extract key findings from the document."
    results = store.search(query, k=top_k)
    passages = [md.get("text", "") for md, _ in results if md.get("text")]
    if not passages:
        return []

    # Build prompt and call LLM
    prompt = _build_prompt(passages, max_findings=max_findings)
    if llm_predict is not None:
        raw = llm_predict(prompt)
    else:
        ollama = get_ollama_predict()
        if ollama is None:
            raise RuntimeError("Local Ollama model not available. Install and run Ollama and ensure qwen2.5:3b is pulled.")
        raw = ollama(prompt)

    bullets = _parse_bullets(raw)
    # Ensure we don't return more than requested
    return bullets[:max_findings]


def extract_findings_from_document(
    path: str,
    llm_predict: Optional[Callable[[str], str]] = None,
    max_findings: int = 6,
    top_k: int = 8,
    temperature: float = 0.0,
) -> List[str]:
    """Load a supported file (pdf, txt, docx, pptx, xml, json, html, csv) and extract key findings.

    All file-based loading MUST use `rag.loader.load_and_split()` to ensure
    binary formats are not decoded as UTF-8.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    from rag.loader import load_and_split

    docs = load_and_split(str(p), chunk_size=800, chunk_overlap=100)
    parts = []
    for d in docs:
        parts.append(getattr(d, "page_content", None) or getattr(d, "text", None) or str(d))
    text = "\n\n".join(parts)

    return extract_findings_from_text(
        text,
        llm_predict=llm_predict,
        max_findings=max_findings,
        top_k=top_k,
        temperature=temperature,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract key findings as bullet points")
    parser.add_argument("path", help="Path to file or '-' to read from stdin")
    parser.add_argument("--max", type=int, default=6, help="Maximum number of findings to return")
    args = parser.parse_args()

    if args.path == "-":
        import sys

        text = sys.stdin.read()
        bullets = extract_findings_from_text(text, max_findings=args.max)
    else:
        bullets = extract_findings_from_document(args.path, max_findings=args.max)

    if not bullets:
        print("No key findings found.")
    else:
        for b in bullets:
            print(f"- {b}")
