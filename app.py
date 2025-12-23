"""Streamlit app for the Agentic RAG pipeline.

Features:
- File upload (PDF, TXT, DOCX, XLSX, CSV) or paste text
- Runs the agentic pipeline: classification -> summarization -> findings
- Uses a cloud LLM (OpenAI) by default when `OPENAI_API_KEY` is set
- Displays document type, a detailed structured summary, and bullet findings

Notes on LLM configuration:
- The app uses the default cloud LLM helper (OpenAI) when available.
- You may also pass a custom `llm_predict` callable to the Controller to use another LLM.
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


from agents.llm import get_llm_predict, is_ollama_available, is_cloud_llm_configured

# Use unified LLM selector by default (prefers Ollama locally, otherwise OpenAI). Both
# LOCAL (Ollama) and CLOUD (OpenAI) modes are supported; selection is runtime-based.


def save_uploaded_file(uploaded) -> str:
    """Save an uploaded file to a temporary path and return the path."""
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        return tmp.name


def main():
    st.set_page_config(page_title="Agentic RAG Demo", layout="wide")

    st.header("Agentic RAG — Document Analysis")

    # Build llm_predict adapter (auto-select Ollama or cloud) and check availability
    try:
        llm_predict = get_llm_predict(temperature=0.0)
        llm_available = True
    except Exception:
        llm_predict = None
        llm_available = False

    # For environments where a local LLM is desired, users may also inject a custom
    # `llm_predict` callable into the Controller. If no LLM is available, disable Run.
    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("### Input")

        if not llm_available:
            # Inform user and explain why LLM features are disabled
            st.warning("No LLM is configured. Install/configure Ollama locally or set OPENAI_API_KEY to enable full functionality; you can still upload files to view loader-only results.")
            st.info("LLM features are disabled in this environment; the Run button is disabled.")

        uploaded = st.file_uploader("Upload a document (Supported: PDF, TXT, DOCX, PPTX, XML, JSON, HTML, CSV, XLSX, CSV)")
        raw_text = st.text_area("Or paste text here (optional)", height=200)
        from agents.classifier import DEFAULT_LABELS
        default_labels = ", ".join(DEFAULT_LABELS)
        labels_input = st.text_input("Allowed labels (comma-separated)", value=default_labels)
        # Disable the Run button when no LLM is available to prevent runtime errors
        run_btn = st.button("Run analysis", disabled=not llm_available)

    with col2:
        st.markdown("### Results")
        label_placeholder = st.empty()
        summary_placeholder = st.empty()
        findings_placeholder = st.empty()

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
