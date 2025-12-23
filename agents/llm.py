"""Unified LLM helper that selects Ollama (local) or a cloud LLM (OpenAI) at runtime.

This module exposes a single, simple API:
- `get_llm_predict(temperature=0.0)` -> returns `callable(prompt) -> str`
- `predict(prompt, temperature=0.0)` -> convenience wrapper

Selection rules:
- If Ollama is available (Python package or CLI), prefer it (LOCAL mode).
- Otherwise, if OpenAI credentials & SDK are present, use OpenAI (CLOUD mode).
- If neither is available, functions raise RuntimeError and callers should
  either provide a custom `llm_predict` or configure an LLM.

Only this module should reference Ollama directly; other project modules
must be LLM-agnostic and call `get_llm_predict()`.
"""

from typing import Callable, Optional
import logging
import os
import subprocess

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Optional cloud (OpenAI) SDK
try:
    import openai
except Exception:  # pragma: no cover - optional dependency
    openai = None

# Optional Ollama Python API (if installed locally)
try:
    import ollama
except Exception:  # pragma: no cover - optional dependency
    ollama = None


def is_ollama_available() -> bool:
    """Return True if Ollama appears available locally (Python package or CLI)."""
    if ollama is not None:
        return True
    # Fallback: check if `ollama` CLI is present
    try:
        subprocess.run(["ollama", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False


def is_cloud_llm_configured() -> bool:
    """Return True if a cloud LLM (OpenAI) is configured and usable."""
    if openai is None:
        return False
    return bool(os.environ.get("OPENAI_API_KEY"))


def get_ollama_predict(model: str = "qwen2.5:3b", temperature: float = 0.0, timeout: int = 180) -> Callable[[str], str]:
    """Return a callable(prompt)->str that uses the Ollama HTTP API for inference.

    This implementation calls POST http://127.0.0.1:11434/api/generate with JSON:
    {"model": model, "prompt": prompt, "stream": false}

    Requirements enforced:
    - Use requests.post
    - endpoint /api/generate
    - send "prompt" field
    - stream=false
    - parse response["response"]
    - timeout >= 180s by default
    - do not swallow exceptions; raise clear RuntimeError with server response text
    """
    if not is_ollama_available():
        raise RuntimeError("Ollama HTTP server not reachable at http://127.0.0.1:11434")

    try:
        import requests
    except Exception as e:  # pragma: no cover - optional dependency
        raise RuntimeError("The 'requests' package is required for Ollama HTTP mode") from e

    url = "http://127.0.0.1:11434/api/generate"

    def _predict(prompt: str) -> str:
        payload = {"model": model, "prompt": prompt, "stream": False}
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
        except Exception as e:
            # Surface transport-level errors clearly
            raise RuntimeError(f"Ollama HTTP request failed: {e}") from e

        # If server returned non-2xx status, include body in error
        if resp.status_code < 200 or resp.status_code >= 300:
            raise RuntimeError(f"Ollama inference failed: status {resp.status_code}: {resp.text}")

        # Parse JSON body
        try:
            j = resp.json()
        except Exception as e:
            raise RuntimeError(f"Ollama returned non-JSON response: {resp.status_code}: {resp.text}") from e

        # Prefer explicit 'response' field
        if "response" in j and isinstance(j["response"], str):
            return j["response"]

        # Fall back to other common fields
        if "output" in j and isinstance(j["output"], str):
            return j["output"]
        if "text" in j and isinstance(j["text"], str):
            return j["text"]

        raise RuntimeError(f"Ollama inference returned unexpected payload: {j}")

    return _predict


def get_default_llm_predict(temperature: float = 0.0) -> Callable[[str], str]:
    """Return a callable that accepts a prompt and returns text via OpenAI.

    Raises RuntimeError if OpenAI is not configured.
    """
    if not is_cloud_llm_configured():
        raise RuntimeError("OpenAI is not configured. Set OPENAI_API_KEY and install openai package.")

    def _predict(prompt: str) -> str:
        # prefer ChatCompletion, fall back to older Completion API
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini" if getattr(openai, "gpt", None) is not None else "gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=3000,
            )
            content = resp.choices[0].message.content
            return content
        except Exception:
            resp = openai.Completion.create(model="gpt-4", prompt=prompt, max_tokens=3000, temperature=temperature)
            return resp.choices[0].text

    return _predict


def get_llm_predict(temperature: float = 0.0, prefer_ollama: bool = True) -> Callable[[str], str]:
    """Return a callable(prompt)->str that selects Ollama (local) or OpenAI (cloud).

    Selection logic:
    - If `prefer_ollama` and Ollama is available, use Ollama.
    - Else if OpenAI configured, use OpenAI.
    - Otherwise raise RuntimeError.

    This function is the single entrypoint that other modules should call.
    """
    if prefer_ollama and is_ollama_available():
        return get_ollama_predict(temperature=temperature)
    if is_cloud_llm_configured():
        return get_default_llm_predict(temperature=temperature)
    raise RuntimeError("No LLM available: install/configure Ollama or set OPENAI_API_KEY")


def predict(prompt: str, temperature: float = 0.0) -> str:
    """Convenience wrapper: synchronous predict using auto-selected backend."""
    fn = get_llm_predict(temperature=temperature)
    return fn(prompt)
