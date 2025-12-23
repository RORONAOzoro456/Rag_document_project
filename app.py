"""Streamlit app for the Agentic RAG pipeline.

Features:
- File upload (PDF, TXT, DOCX) or paste text
- Runs the agentic pipeline: classification -> summarization -> findings
- Uses Ollama's local model `qwen2.5:3b` when available
- Displays document type, summary (<=200 words), and bullet findings

Notes on Ollama integration:
- The app will attempt to use the `ollama` Python client if installed.
- If you prefer a different LLM or remote API, pass a custom `llm_predict`
  callable to the Controller in code or modify the app to use your client.
"""

from __future__ import annotations

import streamlit as st
from pathlib import Path
import tempfile
import logging
from typing import Optional, Callable

from agents.controller import Controller

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


from agents.llm import get_ollama_predict

# Use centralized Ollama helper which enforces local-only policy and logs failures.


def save_uploaded_file(uploaded) -> str:
    """Save an uploaded file to a temporary path and return the path."""
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        return tmp.name


def main():
    st.set_page_config(page_title="Agentic RAG Demo", layout="wide")

    st.header("Agentic RAG — Document Analysis")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("### Input")
        uploaded = st.file_uploader("Upload a document (Supported: PDF, TXT, DOCX, PPTX, XML, JSON, HTML, CSV)")
        raw_text = st.text_area("Or paste text here (optional)", height=200)
        from agents.classifier import DEFAULT_LABELS
        default_labels = ", ".join(DEFAULT_LABELS)
        labels_input = st.text_input("Allowed labels (comma-separated)", value=default_labels)
        run_btn = st.button("Run analysis")

    with col2:
        st.markdown("### Results")
        label_placeholder = st.empty()
        summary_placeholder = st.empty()
        findings_placeholder = st.empty()

    # Build llm_predict adapter for Ollama if present
    llm_predict = get_ollama_predict()

    controller = Controller(labels=[l.strip() for l in labels_input.split(",")], llm_predict=llm_predict)

    if run_btn:
        if not uploaded and not raw_text:
            st.warning("Please upload a file or paste text to analyze.")
            return

        try:
            if uploaded:
                suffix = Path(uploaded.name).suffix.lower()
                supported = {".pdf", ".txt", ".docx", ".pptx", ".xml", ".json", ".html", ".htm", ".csv"}
                if suffix not in supported:
                    st.error(f"Unsupported file type: {suffix}. Supported types: {', '.join(sorted(supported))}")
                    return
                path = save_uploaded_file(uploaded)

                # IMPORTANT: Use load_and_split only for files; the controller
                # will call `process_file()` which itself uses the loader.
                result = controller.process_file(path)
            else:
                # Raw pasted text: process as text (no file IO)
                result = controller.process_text(raw_text)

            # Display results
            lbl = result.get("label")
            if lbl:
                label_placeholder.markdown(f"**Document type:** {lbl}")
            else:
                label_placeholder.markdown("**Document type:** _unknown_")

            summary = result.get("summary")
            if summary:
                summary_placeholder.markdown("### Summary")
                summary_placeholder.write(summary)
            else:
                summary_placeholder.markdown("### Summary\n_No summary available._")

            findings = result.get("findings") or []
            findings_placeholder.markdown("### Key Findings")
            if findings:
                for f in findings:
                    findings_placeholder.markdown(f"- {f}")
            else:
                findings_placeholder.markdown("No key findings found.")

        except Exception as e:
            st.error(f"Processing failed: {e}")


if __name__ == "__main__":
    main()
