"""Orchestration controller for the RAG pipeline.

Architecture overview (what this controller does and why):

- Purpose: provide a single, testable entrypoint that coordinates the
  document-processing agents in this project: classification, summarization,
  and findings extraction. The controller centralizes common concerns such as
  loading document text, passing a consistent LLM interface, and handling
  fallbacks when loaders or LLMs are not available.

- Components and data flow:
  1. Input: a file path (PDF, TXT, DOCX) or raw text.
  2. Classification: runs first to determine the single document label
	 (e.g., "business_report", "invoice", etc.). The classifier uses an
	 LLM and returns exactly one label. The controller provides the full
	 document text to the classifier so the model can use global context.
  3. Summarization: uses RAG retrieval over document chunks and an LLM to
	 produce a concise (<= 200 word) summary that is constrained to the
	 retrieved passages to reduce hallucinations.
  4. Findings extraction: also uses RAG retrieval and an LLM to extract
	 short bullet-point findings.
  5. Output: a structured dict containing the label, summary, and findings.

- Design decisions:
  - The controller favors explicit, deterministic LLM behavior by passing
	the same `llm_predict` callable to all agents when provided, or allowing
	agents to use their LangChain/OpenAI default if not.
  - We try to use the project's loader utilities to read files (PDF/DOCX)
	but gracefully fall back to plain-text reading when necessary.
  - Each step is optional and can be toggled (e.g., run only classification
	or only findings extraction) which makes the controller useful in
	pipelines with conditional routing.

Usage:
	from agents.controller import Controller
	c = Controller(labels=["business_report", "invoice"])  # optional
	result = c.process_file("sample_data/business_report.txt")

The controller is intentionally small and focuses on orchestration rather
than re-implementing agent logic. That keeps responsibilities separated and
makes the system easier to test and extend.
"""

from __future__ import annotations

from typing import Sequence, Optional, Callable, Dict, Any, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Import agent modules (these are in the repository)
from agents.classifier import classify_document
from agents.summarizer import summarize_document
from agents.findings import extract_findings_from_document

# Try to use the project's loader to read file text when needed. We import
# lazily inside methods to avoid circular import issues.


class Controller:
	"""Orchestrates classification, summarization, and findings extraction.

	Parameters:
		labels: optional list of allowed labels for classification. If None,
			callers must pass labels to `process_text`/`process_file`.
		llm_predict: optional callable(prompt: str) -> str. If provided it
			will be forwarded to the underlying agents to control LLM calls
			(useful for testing or using a custom LLM wrapper).
		temperature: temperature passed to agents when they support it.
	"""

	def __init__(
		self,
		labels: Optional[Sequence[str]] = None,
		llm_predict: Optional[Callable[[str], str]] = None,
		temperature: float = 0.0,
	) -> None:
		self.labels = list(labels) if labels is not None else None
		self.llm_predict = llm_predict
		self.temperature = temperature

	def _read_file_text(self, path: str) -> str:
		"""Attempt to read file contents.

		Strategy:
		- Prefer using `rag.loader.load_and_split` when available so PDFs/DOCX
		  are parsed correctly.
		- Fallback to reading as UTF-8 text for TXT files or when the loader
		  is unavailable.
		"""
		p = Path(path)
		if not p.exists():
			raise FileNotFoundError(f"File not found: {p}")

		# Use rag.loader.load_and_split to obtain parsed chunks and join them.
		# IMPORTANT: Do NOT attempt to read binary formats as text.
		from rag.loader import load_and_split

		docs = load_and_split(str(p), chunk_size=800, chunk_overlap=100)
		parts: List[str] = []
		for d in docs:
			parts.append(getattr(d, "page_content", None) or getattr(d, "text", None) or str(d))
		return "\n\n".join(parts)

	def process_file(
		self,
		path: str,
		labels: Optional[Sequence[str]] = None,
		run_classify: bool = True,
		run_summarize: bool = True,
		run_findings: bool = True,
		summarize_max_words: int = 200,
		summary_depth: str = "medium",
	) -> Dict[str, Any]:
		"""Process a file through the pipeline and return structured results.

		Returns a dict with keys: 'label' (optional), 'summary' (optional),
		and 'findings' (list of bullet strings, optional).
		"""
		text = self._read_file_text(path)
		return self.process_text(
			text,
			labels=labels,
			run_classify=run_classify,
			run_summarize=run_summarize,
			run_findings=run_findings,
			summarize_max_words=summarize_max_words,
			summary_depth=summary_depth,
		)

	def process_text(
		self,
		text: str,
		labels: Optional[Sequence[str]] = None,
		run_classify: bool = True,
		run_summarize: bool = True,
		run_findings: bool = True,
		summarize_max_words: int = 200,
		summary_depth: str = "medium",
		source_path: Optional[str] = None,
	) -> Dict[str, Any]:
		"""Process raw text.

		Args:
			text: document text to process.
			labels: allowed labels for classification (overrides controller labels).
			run_classify/run_summarize/run_findings: toggle individual steps.
			summarize_max_words: summary word limit.
			source_path: optional path used when delegating to file-based helpers.
		"""
		results: Dict[str, Any] = {"label": None, "summary": None, "findings": []}

		effective_labels = list(labels) if labels is not None else self.labels
		if run_classify:
			if not effective_labels:
				raise ValueError("No labels provided for classification")
			# Classifier expects the document text
			label = classify_document(
				text,
				effective_labels,
				llm_predict=self.llm_predict,
				temperature=self.temperature,
			)
			results["label"] = label

		if run_summarize:
			# summarizer has a file-based helper; if source_path is provided
			# prefer it so the summarizer can load using the project's loader
			try:
				# Create a progress callback that uses Streamlit if available so the UI
				# receives live updates without requiring changes to app.py.
				progress_cb = None
				try:
					import streamlit as st
					progress_bar = st.progress(0)
					status = st.empty()
					def _progress(current: int, total: int) -> None:
						# Update simple progress UI: chunk number and percentage
						pct = int((current / total) * 100)
						progress_bar.progress(pct)
						status.text(f"Summarizing chunk {current}/{total} ({pct}%)")
					progress_cb = _progress
				except Exception:
					# Streamlit not available or UI context missing; continue without UI
					progress_cb = None

				if source_path:
					summary = summarize_document(
						source_path,
						llm_predict=self.llm_predict,
						depth=summary_depth,
						temperature=self.temperature,
						chunk_size=4000,
						progress_callback=progress_cb,
					)
				else:
					# Use text-based summarizer with progress updates
					from agents.summarizer import summarize_text

					summary = summarize_text(
						text,
						llm_predict=self.llm_predict,
						depth=summary_depth,
						chunk_size=4000,
						progress_callback=progress_cb,
						temperature=self.temperature,
					)
				results["summary"] = summary
			except Exception as e:  # pragma: no cover - runtime errors depend on env
				logger.exception("Summarization failed: %s", e)
				results["summary"] = None

		if run_findings:
			try:
				if source_path:
					findings = extract_findings_from_document(
						source_path,
						llm_predict=self.llm_predict,
						temperature=self.temperature,
					)
				else:
					findings = extract_findings_from_text(
						text,
						llm_predict=self.llm_predict,
						temperature=self.temperature,
					)
				results["findings"] = findings
			except Exception as e:  # pragma: no cover
				logger.exception("Findings extraction failed: %s", e)
				results["findings"] = []

		return results


def extract_findings_from_text(text: str, **kwargs):
	"""Local import helper to avoid circular import at module load time."""
	from agents.findings import extract_findings_from_text as _f

	return _f(text, **kwargs)


if __name__ == "__main__":
	# Small CLI to run the controller on a single file and print results.
	import argparse
	import json

	parser = argparse.ArgumentParser(description="Run controller on a document")
	parser.add_argument("path", help="Path to document (pdf, txt, docx)")
	parser.add_argument("--labels", nargs="+", help="Allowed labels for classification", required=True)
	args = parser.parse_args()

	c = Controller(labels=args.labels)
	out = c.process_file(args.path)
	print(json.dumps(out, indent=2, ensure_ascii=False))

