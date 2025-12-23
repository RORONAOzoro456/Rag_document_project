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

from rag.loader import load_and_split_documents as load_and_split
from rag.vectorstore import FaissVectorStore

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Support hierarchical summarization and cloud LLM defaults (OpenAI)
try:
    import openai
except Exception:  # pragma: no cover - optional dependency
    openai = None


def _default_cloud_llm_predict(prompt: str, temperature: float = 0.0) -> str:
    """Default LLM wrapper using OpenAI's chat completion API.

    Requires OPENAI_API_KEY in environment. If `openai` is not installed or
    API key is not set, callers should pass a custom `llm_predict`.
    """
    if openai is None:
        raise RuntimeError("OpenAI SDK not installed. Install openai package or provide llm_predict.")
    # Prefer GPT-4 family if available; fall back to gpt-3.5-turbo
    model = "gpt-4o-mini" if getattr(openai, "gpt", None) is not None else "gpt-4"
    # Use Chat Completions if available
    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=3000,
        )
        content = resp.choices[0].message.content
        return content
    except Exception as e:  # pragma: no cover - runtime env specific
        # Try the older completion API as fallback
        resp = openai.Completion.create(model="gpt-4", prompt=prompt, max_tokens=3000, temperature=temperature)
        return resp.choices[0].text


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


def _summarize_chunk(
    chunk: str,
    llm_predict: Callable[[str], str],
    depth: str = "medium",
    temperature: float = 0.0,
) -> str:
    """Produce a focused summary for a single chunk requesting key concepts, explanations, examples, conclusions."""
    depth_map = {
        "short": "Keep the chunk summary concise (1-3 short sentences).",
        "medium": "Provide a detailed paragraph with key concepts and brief explanations (3-6 sentences).",
        "detailed": "Provide a detailed paragraph emphasizing concepts, explanations and examples if present (4-8 sentences).",
    }
    instruction = depth_map.get(depth, depth_map["medium"])
    prompt = (
        "You are a helpful summarization assistant. For the provided text, produce a focused summary that includes:\n"
        "- Key concepts and important facts.\n"
        "- Brief explanations of those concepts.\n"
        "- Examples if present in the text.\n"
        "- A short concluding sentence.\n\n"
        f"{instruction}\n\n"
        "TEXT:\n"
        f"{chunk}\n\n"
        "Provide a short structured paragraph as the chunk summary."
    )
    return llm_predict(prompt)


def _synthesize_section(
    summaries: Sequence[str],
    llm_predict: Callable[[str], str],
    depth: str = "medium",
    temperature: float = 0.0,
) -> str:
    """Combine several chunk summaries into a section-level summary."""
    prompt = (
        "You are an assistant that combines several small summaries into a coherent section summary."
        " Use ONLY the provided summaries to synthesize a single coherent paragraph that captures the major points and their relationships."
        " Be explicit about key conclusions and any examples.\n\n"
        "INPUT SUMMARIES:\n"
        f"{chr(10).join(f'SUMMARY {i+1}: {s}' for i, s in enumerate(summaries))}\n\n"
        "Produce a clear, well-structured paragraph representing the section."
    )
    return llm_predict(prompt)


def _final_synthesis(
    section_summaries: Sequence[str],
    llm_predict: Callable[[str], str],
    depth: str = "medium",
    temperature: float = 0.0,
) -> str:
    """Produce the final multi-section, detailed synthesis proportional to document size."""
    depth_settings = {
        "short": {"min_paragraphs": 2, "max_paragraphs": 4},
        "medium": {"min_paragraphs": 3, "max_paragraphs": 6},
        "detailed": {"min_paragraphs": 6, "max_paragraphs": 12},
    }
    ds = depth_settings.get(depth, depth_settings["medium"])

    prompt = (
        "You are an expert writer preparing an executive-style document summary."
        " Use ONLY the provided SECTION blocks to write a multi-section summary that reads like an executive report."
        " Each major section should have a short heading and 1-3 paragraphs. Emphasize key concepts, explanations, examples (if any), and actionable conclusions."
        " The final summary should be proportional to document length: for large inputs, produce multiple paragraphs (at least) and maintain clear structure.\n\n"
        "SECTIONS:\n"
        f"{chr(10).join(f'SECTION {i+1}: {s}' for i, s in enumerate(section_summaries))}\n\n"
        f"Write a final multi-section summary. Aim for {ds['min_paragraphs']} to {ds['max_paragraphs']} paragraphs overall, with headings for each major section when helpful."
    )
    return llm_predict(prompt)


def _ensure_llm(llm_predict: Optional[Callable[[str], str]], temperature: float = 0.0) -> Callable[[str], str]:
    """Return a callable llm_predict; prefer provided callable, otherwise use unified LLM selector.

    This delegates to `agents.llm.get_llm_predict` which selects Ollama (local)
    when available or OpenAI when running in cloud environments.
    """
    if llm_predict is not None:
        return lambda p: llm_predict(p)

    # Defer to agents.llm for auto-selection between Ollama and OpenAI
    try:
        from agents.llm import get_llm_predict

        return get_llm_predict(temperature=temperature)
    except Exception as e:
        raise RuntimeError(
            "No LLM available: install/configure Ollama or set OPENAI_API_KEY, or pass llm_predict to summarizer."
        ) from e


def summarize_text(
    text: str,
    llm_predict: Optional[Callable[[str], str]] = None,
    depth: str = "medium",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    temperature: float = 0.0,
) -> str:
    """Hierarchical summarization of raw text using map-reduce style steps.

    Steps:
    - Split the text into chunks
    - Map: produce a chunk-level summary for each chunk
    - Reduce: group chunk summaries into section summaries
    - Synthesize: create final multi-section summary
    """
    from rag.loader import get_text_splitter

    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)

    if not chunks:
        return "Insufficient information to produce a reliable summary."

    llm = _ensure_llm(llm_predict, temperature=temperature)

    # Map: summarize each chunk
    chunk_summaries: List[str] = []
    for c in chunks:
        s = _summarize_chunk(c, llm, depth=depth, temperature=temperature)
        chunk_summaries.append(s.strip())

    # Determine grouping for section synthesis based on number of chunks
    num_chunks = max(1, len(chunk_summaries))
    # target number of sections scales with document size and depth
    if depth == "short":
        target_sections = min(6, max(1, num_chunks // 8))
    elif depth == "detailed":
        target_sections = min(12, max(2, num_chunks // 4))
    else:
        target_sections = min(8, max(1, num_chunks // 6))
    target_sections = max(1, target_sections)

    # Group sequentially
    group_size = max(1, -(-num_chunks // target_sections))  # ceil division
    section_summaries: List[str] = []
    for i in range(0, num_chunks, group_size):
        group = chunk_summaries[i : i + group_size]
        sec = _synthesize_section(group, llm, depth=depth, temperature=temperature)
        section_summaries.append(sec.strip())

    # Final synthesis
    final = _final_synthesis(section_summaries, llm, depth=depth, temperature=temperature)
    return final


def summarize_document(
    path: str,
    llm_predict: Optional[Callable[[str], str]] = None,
    depth: str = "medium",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    temperature: float = 0.0,
) -> str:
    """Load a file and perform hierarchical summarization.

    This function loads and chunks the document using `rag.loader.load_and_split_documents`
    and then applies the hierarchical summarization pipeline above.
    """
    docs = load_and_split(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Extract texts from Document objects
    chunks: List[str] = []
    for d in docs:
        content = getattr(d, "page_content", None) or getattr(d, "text", None) or str(d)
        if content:
            chunks.append(content)

    if not chunks:
        return "Insufficient information to produce a reliable summary."

    # Merge small chunks where appropriate to reduce number of summaries
    # (keep them as-is otherwise)
    combined_text = "\n\n".join(chunks)

    return summarize_text(
        combined_text,
        llm_predict=llm_predict,
        depth=depth,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        temperature=temperature,
    )





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
