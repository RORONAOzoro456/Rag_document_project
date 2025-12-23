"""Document type classifier backed by an LLM.

This module provides a simple helper to classify a piece of document text
into exactly one label using a language model. The function `classify_document`
returns a single label (string) chosen from a provided list.

Prompting logic (why it's designed this way):
- We provide the LLM an explicit, constrained instruction to "Return only a
  single label". This reduces unexpected chattiness and makes parsing
  deterministic for downstream code.
- We give the full list of allowed labels. This prevents the model from
  inventing labels or synonyms, and keeps the output within the allowed set.
- We set the LLM to be deterministic (temperature=0) when possible to
  encourage consistent, repeatable decisions.
- We instruct the model to prefer the most specific label when multiple
  labels could apply. This reduces ambiguity and makes the classifier
  useful for routing/metadata tagging.

The classifier is written to work with LangChain LLMs (preferred) or any
callable `llm_predict(prompt: str) -> str` provided by the caller. If no
LLM is available, the function raises a helpful error.
"""

from __future__ import annotations

from typing import List, Sequence, Optional, Callable
import logging
from difflib import get_close_matches

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from agents.llm import get_ollama_predict

# Default labels recommended for the project
DEFAULT_LABELS = [
	"Legal Document",
	"Certificate",
	"Invoice",
	"Transactional Record",
	"Configuration File",
	"Presentation",
	"Research Paper",
	"Business Report",
	"Essay / General Text",
]


def _build_prompt(labels: Sequence[str], document: str) -> str:
	"""Construct the prompt string given labels and the document.

	The prompt explicitly instructs the LLM to output ONLY a single label
	exactly matching one of the provided labels. This explicitness helps
	reduce hallucinated or verbose outputs.
	"""
	labels_block = "\n".join(f"- {l}" for l in labels)
	prompt = (
		"You are a concise document classifier.\n"
		"Choose exactly ONE label from the list of allowed labels that best describes the document below.\n"
		"Output ONLY the single label (exactly as shown in the list), with no extra text, punctuation, or explanation.\n"
		"If multiple labels apply, select the most specific one.\n\n"
		"ALLOWED LABELS:\n"
		f"{labels_block}\n\n"
		"DOCUMENT:\n"
		f"{document}\n\n"
		"Return only the chosen label:" 
	)
	return prompt


def classify_document(
	document: str,
	labels: Sequence[str],
	llm_predict: Optional[Callable[[str], str]] = None,
	temperature: float = 0.0,
) -> str:
	"""Classify `document` into exactly one label from `labels`.

	Args:
		document: The text to classify.
		labels: Sequence of allowed label strings. The returned label will be
			matched against this list.
		llm_predict: Optional callable that accepts a prompt string and
			returns the model's raw text output. If omitted and LangChain
			is available, an OpenAI LLM with temperature=0 will be used.
		temperature: Desired temperature for deterministic output where
			supported. Default is 0.0.

	Returns:
		A single label (one of the supplied `labels`).

	Raises:
		RuntimeError: If no LLM is available and no llm_predict was provided.
		ValueError: If labels is empty or document is empty.
	"""
	if not labels:
		raise ValueError("labels must be a non-empty sequence")
	if not document:
		raise ValueError("document must be a non-empty string")

	prompt = _build_prompt(labels, document)

	# If user provided a callable llm_predict, use it.
	if llm_predict is not None:
		raw = llm_predict(prompt)
	else:
		# Enforce local-only LLM usage: try Ollama, otherwise raise error
		ollama = get_ollama_predict()
		if ollama is None:
			raise RuntimeError("Local Ollama model not available. Install and run Ollama and ensure qwen2.5:3b is pulled.")
		raw = ollama(prompt)

	if raw is None:
		raise RuntimeError("LLM produced no output")

	candidate = raw.strip().splitlines()[0].strip()

	# If the model returned an exact match (case-sensitive), accept it
	if candidate in labels:
		return candidate

	# Try case-insensitive match
	lower_map = {l.lower(): l for l in labels}
	if candidate.lower() in lower_map:
		return lower_map[candidate.lower()]

	# As a last resort, try fuzzy matching the returned token to one of the labels
	close = get_close_matches(candidate, labels, n=1, cutoff=0.6)
	if close:
		return close[0]

	# If the LLM didn't follow instructions, try to choose the best label by
	# checking which label string appears most in the document (simple heuristic)
	doc_lower = document.lower()
	best_label = max(labels, key=lambda l: doc_lower.count(l.lower()))
	logger.debug(
		"LLM output (%s) did not match allowed labels; falling back to heuristic: %s",
		candidate,
		best_label,
	)
	return best_label


if __name__ == "__main__":
	# Small manual test/CLI
	import argparse

	parser = argparse.ArgumentParser(description="Classify a document into one label using an LLM.")
	parser.add_argument("--file", type=str, help="Path to a text file to classify")
	parser.add_argument("--labels", type=str, nargs="+", required=True, help="Allowed labels")
	args = parser.parse_args()

	if not args.file:
		print("Please provide --file PATH and --labels LABEL1 LABEL2 ...")
	else:
		with open(args.file, "r", encoding="utf-8") as fh:
			text = fh.read()
		try:
			label = classify_document(text, args.labels)
			print(label)
		except Exception as e:
			print(f"Error: {e}")

