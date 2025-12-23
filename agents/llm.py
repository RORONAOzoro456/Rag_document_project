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


def get_ollama_predict(model: str = "llama2", temperature: float = 0.0) -> Callable[[str], str]:
    """Return a callable(prompt)->str that uses Ollama for inference.

    Tries multiple ways to invoke Ollama (Python package or `ollama` CLI).
    Raises RuntimeError if Ollama is not available or an invocation fails.
    """
    if not is_ollama_available():
        raise RuntimeError("Ollama is not available on this system")

    def _predict(prompt: str) -> str:
        # Try Python package API first (if present)
        if ollama is not None:
            try:
                # Modern Ollama client may provide a `generate` function or a client class
                if hasattr(ollama, "Ollama"):
                    client = ollama.Ollama()
                    out = client.generate(model=model, prompt=prompt, temperature=temperature)
                    # Try common attributes
                    if isinstance(out, dict):
                        return out.get("output") or out.get("text") or str(out)
                    return getattr(out, "text", getattr(out, "output", str(out)))
                if hasattr(ollama, "generate"):
                    out = ollama.generate(model=model, prompt=prompt, temperature=temperature)
                    if isinstance(out, dict):
                        return out.get("output") or out.get("text") or str(out)
                    return getattr(out, "text", getattr(out, "output", str(out)))
            except Exception as e:
                logger.debug("Ollama Python API failed: %s", e)
                # Fall through to CLI attempt

        # Fallback to calling `ollama` CLI (requires Ollama binary in PATH)
        try:
            proc = subprocess.run(["ollama", "run", model, "--max-tokens", "3000"], input=prompt.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            out = proc.stdout.decode("utf-8").strip()
            return out
        except Exception as e:
            logger.error("Ollama invocation failed: %s", e)
            raise RuntimeError("Ollama inference failed")

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
