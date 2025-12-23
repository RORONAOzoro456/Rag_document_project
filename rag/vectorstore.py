"""FAISS-backed vector store using Sentence-Transformers.

This module provides a small, production-oriented wrapper around FAISS
to build and query a vector store using the SentenceTransformers
model "all-MiniLM-L6-v2".

Why FAISS?
- FAISS is a highly-optimized library for similarity search. It provides
  efficient nearest-neighbor search on CPU and GPU, supports many index
  types (exact and approximate), and scales to large collections. For RAG
  systems where you need fast retrieval over thousands to millions of
  embeddings, FAISS is a practical, well-supported choice.

The FaissVectorStore class below keeps an in-memory FAISS index and a
parallel Python list of metadata objects (one per vector). It supports
adding documents, querying by similarity, and saving/loading to disk.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Sequence, Tuple, Union
from pathlib import Path
import logging
import pickle

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - informative import error at runtime
    SentenceTransformer = None  # type: ignore

try:
    import faiss
except Exception:  # pragma: no cover
    faiss = None  # type: ignore

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class FaissVectorStore:
    """Simple FAISS-backed vector store.

    Attributes:
        model_name: sentence-transformers model name.
        model: SentenceTransformer instance.
        index: faiss.Index (IndexFlatIP used for cosine search on normalized vectors).
        metadatas: list of metadata dicts aligned with vectors in index.
        dim: embedding dimension.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence_transformers not installed. Install with: pip install sentence-transformers"
            )
        if faiss is None:
            raise RuntimeError("faiss is not installed. Install faiss (faiss-cpu) to use this module.")

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        # Build an exact inner-product index. We'll normalize vectors to use
        # cosine-similarity equivalently with inner product.
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dim)
        # metadatas keeps user-provided metadata (e.g., source, chunk_id, text)
        self.metadatas: List[Dict[str, Any]] = []

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a sequence of texts and return L2-normalized float32 vectors."""
        embs = self.model.encode(list(texts), convert_to_numpy=True)
        # Convert to float32 for FAISS
        embs = embs.astype("float32")
        # Normalize to unit length for cosine similarity using inner product
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        # avoid division by zero
        norms[norms == 0.0] = 1.0
        embs = embs / norms
        return embs

    def add_documents(self, texts: Sequence[str], metadatas: Optional[Sequence[Dict[str, Any]]] = None) -> None:
        """Add documents (texts) with optional metadata to the vector store.

        Args:
            texts: sequence of strings to embed and add.
            metadatas: optional sequence of metadata dicts aligned to texts.
        """
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError("Length of metadatas must match length of texts")

        embs = self._embed(texts)
        self.index.add(embs)
        if metadatas is None:
            # At minimum store the text so callers can see it in results
            for t in texts:
                self.metadatas.append({"text": t})
        else:
            for md in metadatas:
                self.metadatas.append(dict(md))

    def search(self, query: str, k: int = 4) -> List[Tuple[Dict[str, Any], float]]:
        """Return top-k nearest metadata and similarity score for the query.

        The scores returned are cosine-similarity in [-1, 1].
        """
        q_emb = self._embed([query])
        # FAISS returns distances; with IndexFlatIP and normalized vectors, the
        # distances are inner products == cosine similarity.
        D, I = self.index.search(q_emb, k)
        D = D[0]  # shape (k,)
        I = I[0]  # shape (k,)

        results: List[Tuple[Dict[str, Any], float]] = []
        for idx, score in zip(I, D):
            if idx < 0 or idx >= len(self.metadatas):
                continue
            results.append((self.metadatas[idx], float(score)))
        return results

    def save(self, folder: Union[str, Path]) -> None:
        """Persist FAISS index and metadata to disk.

        The index is saved using faiss.write_index and the metadatas are
        pickled to metadata.pkl in the same folder.
        """
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(folder / "index.faiss"))
        with open(folder / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadatas, f)
        # Also store model_name for reproducibility
        with open(folder / "store_info.pkl", "wb") as f:
            pickle.dump({"model_name": self.model_name, "dim": self.dim}, f)

    @classmethod
    def load(cls, folder: Union[str, Path]) -> "FaissVectorStore":
        """Load a persisted FaissVectorStore from disk.

        This will re-instantiate the sentence-transformers model, load the
        FAISS index, and restore metadata mapping.
        """
        folder = Path(folder)
        if not (folder / "index.faiss").exists():
            raise FileNotFoundError(f"No FAISS index found in {folder}")

        with open(folder / "store_info.pkl", "rb") as f:
            info = pickle.load(f)

        obj = cls(model_name=info.get("model_name", "all-MiniLM-L6-v2"))
        # load index and replace the empty index created in __init__
        obj.index = faiss.read_index(str(folder / "index.faiss"))
        with open(folder / "metadata.pkl", "rb") as f:
            obj.metadatas = pickle.load(f)
        return obj


def from_documents(
    documents: Sequence[Union[str, Dict[str, Any]]],
    model_name: str = "all-MiniLM-L6-v2",
) -> FaissVectorStore:
    """Convenience helper: build a FaissVectorStore from documents.

    documents can be either raw strings or metadata dicts with a 'text'
    key containing the text. This helper extracts texts and metadatas and
    returns a populated FaissVectorStore.
    """
    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    for d in documents:
        if isinstance(d, str):
            texts.append(d)
            metadatas.append({"text": d})
        elif isinstance(d, dict):
            text = d.get("text")
            if not text:
                raise ValueError("Document dicts must contain a 'text' key")
            texts.append(text)
            # Copy metadata without the text to avoid duplication
            md = dict(d)
            metadatas.append(md)
        else:
            raise TypeError("documents must be str or dict with 'text' key")

    store = FaissVectorStore(model_name=model_name)
    store.add_documents(texts, metadatas)
    return store


__all__ = ["FaissVectorStore", "from_documents"]
