"""LLM helpers for local-only model usage (Ollama).

This module centralizes how the repository acquires a callable LLM
predict function that sends prompts to a local Ollama model. The
project enforces a local-only policy: if Ollama is not available the
helpers raise an explicit error instructing the user to install and
run Ollama rather than falling back to cloud APIs.
"""

from typing import Callable, Optional
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def get_ollama_predict() -> Optional[Callable[[str], str]]:
    """Return a prompt -> text callable that uses Ollama's local model.

    This uses the newer `ollama.Client` API (the package exposes a `Client`
    class which provides `generate(...)`). If the client is not importable
    or a runtime issue occurs, the function returns None so callers can
    raise helpful errors.
    """
    try:
        from ollama import Client

        client = Client()

        def _predict(prompt: str) -> str:
            out = client.generate(model="qwen2.5:3b", prompt=prompt)
            # The library returns a typed GenerateResponse; extract the
            # textual response if present, otherwise fall back to dict/string
            if hasattr(out, "response") and out.response is not None:
                return out.response if isinstance(out.response, str) else str(out.response)
            if isinstance(out, dict):
                return out.get("text") or out.get("result") or str(out)
            return str(out)

        return _predict
    except Exception as e:  # pragma: no cover - depends on user's environment
        logger.debug("Ollama not available: %s", e)
        return None
