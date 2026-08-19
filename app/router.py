"""Model → backend routing."""
from __future__ import annotations

import logging
from typing import Dict, List

from fastapi import HTTPException

from app.backends import LLMBackend
from app.backends.messages_http import MessagesHTTPBackend
from app.backends.openai_http import OpenAICompletionsBackend
from app.config import settings

logger = logging.getLogger(__name__)


def _parse_model_map(models_str: str) -> Dict[str, str]:
    """Parse comma-separated "client-id:upstream-id" or bare "model-id" entries."""
    result: Dict[str, str] = {}
    for entry in models_str.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" in entry:
            client_id, upstream_id = entry.split(":", 1)
            result[client_id.strip()] = upstream_id.strip()
        else:
            result[entry] = entry
    return result


def _make_backend(
    name: str,
    base_url: str,
    api_key: str,
    model_map: Dict[str, str],
    fmt: str,
) -> LLMBackend:
    """Build a backend speaking the requested upstream wire format."""
    if fmt == "openai":
        if base_url.rstrip("/").endswith("/anthropic"):
            logger.warning(
                f"[router] {name}: FORMAT=openai but BASE_URL ends with /anthropic — "
                "use the OpenAI-style root (e.g. https://api.deepseek.com)"
            )
        return OpenAICompletionsBackend(name=name, base_url=base_url, api_key=api_key, model_map=model_map)
    return MessagesHTTPBackend(name=name, base_url=base_url, api_key=api_key, model_map=model_map)


def _build_deepseek_backend() -> "LLMBackend | None":
    if not settings.deepseek_api_key:
        logger.info("[router] DEEPSEEK_API_KEY not set — DeepSeek backend disabled")
        return None
    model_map = _parse_model_map(settings.deepseek_models)
    return _make_backend(
        "deepseek",
        settings.deepseek_base_url,
        settings.deepseek_api_key,
        model_map,
        settings.deepseek_format,
    )


def _build_extra_backend() -> "LLMBackend | None":
    if not settings.extra_backend_api_key or not settings.extra_backend_base_url:
        return None
    if not settings.extra_backend_models:
        logger.info("[router] EXTRA_BACKEND_MODELS not set — extra backend disabled")
        return None
    model_map = _parse_model_map(settings.extra_backend_models)
    name = settings.extra_backend_name or "extra"
    return _make_backend(
        name,
        settings.extra_backend_base_url,
        settings.extra_backend_api_key,
        model_map,
        settings.extra_backend_format,
    )


_deepseek_backend = _build_deepseek_backend()
_extra_backend = _build_extra_backend()

MODEL_REGISTRY: Dict[str, LLMBackend] = {}


def _register(backend: "LLMBackend | None") -> None:
    if backend is None:
        return
    for model_id in backend.model_map:
        if model_id in MODEL_REGISTRY:
            prev = MODEL_REGISTRY[model_id]
            logger.warning(f"[router] model {model_id!r} from {prev.name!r} overridden by {backend.name!r}")
        MODEL_REGISTRY[model_id] = backend


_register(_deepseek_backend)
_register(_extra_backend)


def list_models() -> List[str]:
    return list(MODEL_REGISTRY.keys())


def select_backend(client_model: str) -> LLMBackend:
    backend = MODEL_REGISTRY.get(client_model)
    if backend is None:
        raise HTTPException(
            status_code=404,
            detail={
                "type": "not_found_error",
                "message": f"model '{client_model}' is not available on this proxy",
            },
        )
    return backend
