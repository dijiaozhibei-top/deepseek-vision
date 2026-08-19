"""HTTP-based Anthropic Messages-compatible backend.

Handles upstreams that speak the Anthropic Messages API (DeepSeek, or any
other provider with an Anthropic-compatible endpoint). The agentic loops in
web_search / web_fetch use this backend transparently, so those models gain
web_search + web_fetch capability via the proxy middleware.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, Optional

import httpx
from fastapi import HTTPException

from app.backends import LLMBackend
from app.config import settings
from app.schemas import MessageRequest, MessageResponse, Usage

logger = logging.getLogger(__name__)


def _build_body(request: MessageRequest, upstream_model_id: str, stream: bool) -> Dict[str, Any]:
    body = request.model_dump(exclude_none=True)
    body["model"] = upstream_model_id
    body["stream"] = stream

    if upstream_model_id.startswith("deepseek-"):
        # Respect a client-specified effort; default to max so DeepSeek's
        # adaptive thinking stays fully enabled unless the caller opts out.
        output_config = body.get("output_config")
        if not isinstance(output_config, dict) or not isinstance(output_config.get("effort"), str):
            merged = dict(output_config) if isinstance(output_config, dict) else {}
            merged["effort"] = "max"
            body["output_config"] = merged
        # DeepSeek rejects `thinking` when reasoning_effort / output_config.effort is set.
        body.pop("thinking", None)
        # deepseek-reasoner rejects tool_choice.
        body.pop("tool_choice", None)

    return body


def _parse_response(response_body: Dict[str, Any], client_model: str, message_id: str) -> MessageResponse:
    usage_data = response_body.get("usage", {}) or {}
    usage = Usage(
        input_tokens=usage_data.get("input_tokens", 0) or 0,
        output_tokens=usage_data.get("output_tokens", 0) or 0,
        cache_creation_input_tokens=usage_data.get("cache_creation_input_tokens"),
        cache_read_input_tokens=usage_data.get("cache_read_input_tokens"),
    )
    return MessageResponse(
        id=message_id,
        content=response_body.get("content", []) or [],
        model=client_model,
        stop_reason=response_body.get("stop_reason"),
        stop_sequence=response_body.get("stop_sequence"),
        usage=usage,
    )


def _raise_for_http_error(resp: httpx.Response, backend_name: str) -> None:
    try:
        body = resp.json()
    except ValueError:
        body = {"error": {"type": "api_error", "message": resp.text[:500]}}
    err = body.get("error") if isinstance(body, dict) else None
    if not isinstance(err, dict):
        err = {"type": "api_error", "message": str(body)[:500]}
    logger.error(f"[{backend_name}] upstream {resp.status_code}: {err}")
    raise HTTPException(
        status_code=resp.status_code,
        detail={"type": err.get("type", "api_error"), "message": err.get("message", "upstream error")},
    )


class MessagesHTTPBackend(LLMBackend):
    """Generic Anthropic Messages-compatible HTTP backend."""

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str,
        model_map: Dict[str, str],
        anthropic_version: str = "2023-06-01",
    ) -> None:
        self.name = name
        self.model_map = dict(model_map)
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._anthropic_version = anthropic_version

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(float(settings.upstream_timeout), connect=10.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=50),
        )
        self._stream_client = httpx.AsyncClient(
            timeout=httpx.Timeout(float(settings.upstream_stream_timeout), connect=10.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=50),
        )

    def _headers(self, anthropic_beta: Optional[str], stream: bool) -> Dict[str, str]:
        h: Dict[str, str] = {
            "x-api-key": self._api_key,
            "anthropic-version": self._anthropic_version,
            "content-type": "application/json",
            "accept": "text/event-stream" if stream else "application/json",
        }
        if anthropic_beta:
            h["anthropic-beta"] = anthropic_beta
        return h

    async def invoke(
        self,
        request: MessageRequest,
        request_id: str,
        anthropic_beta: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> MessageResponse:
        upstream_model_id = self.resolve_model_id(request.model)
        body = _build_body(request, upstream_model_id, stream=False)
        headers = self._headers(anthropic_beta, stream=False)

        if meta is not None:
            meta["model_id"] = upstream_model_id
            meta["request_size_bytes"] = len(json.dumps(body).encode("utf-8"))

        if settings.debug_upstream:
            logger.info(f"[{self.name}] Request body: {json.dumps(body, ensure_ascii=False)[:2000]}")

        try:
            resp = await self._client.post(
                f"{self._base_url}/v1/messages",
                headers=headers,
                json=body,
            )
        except httpx.TimeoutException as e:
            raise HTTPException(status_code=408, detail={"type": "timeout_error", "message": str(e)})
        except httpx.HTTPError as e:
            logger.error(f"[{self.name}] HTTP error: {e}")
            raise HTTPException(status_code=502, detail={"type": "api_error", "message": str(e)})

        if resp.status_code >= 400:
            _raise_for_http_error(resp, self.name)

        response_body = resp.json()
        if meta is not None:
            meta["response_body"] = response_body
        return _parse_response(response_body, request.model, request_id)

    async def stream(
        self,
        request: MessageRequest,
        request_id: str,
        anthropic_beta: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[str, None]:
        upstream_model_id = self.resolve_model_id(request.model)
        body = _build_body(request, upstream_model_id, stream=True)
        headers = self._headers(anthropic_beta, stream=True)

        if meta is not None:
            meta["model_id"] = upstream_model_id

        yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"

        event_queue: asyncio.Queue = asyncio.Queue(maxsize=1024)
        DONE = object()

        async def _reader():
            try:
                async with self._stream_client.stream(
                    "POST",
                    f"{self._base_url}/v1/messages",
                    headers=headers,
                    json=body,
                ) as resp:
                    if resp.status_code >= 400:
                        await resp.aread()
                        try:
                            _raise_for_http_error(resp, self.name)
                        except HTTPException as http_exc:
                            detail = http_exc.detail if isinstance(http_exc.detail, dict) else {"type": "api_error", "message": str(http_exc.detail)}
                            await event_queue.put(f"event: error\ndata: {json.dumps({'type': 'error', 'error': detail})}\n\n")
                        return

                    buf = ""
                    async for chunk in resp.aiter_text():
                        if not chunk:
                            continue
                        buf += chunk
                        while "\n\n" in buf:
                            event_block, buf = buf.split("\n\n", 1)
                            if event_block.strip():
                                await event_queue.put(event_block + "\n\n")
                    if buf.strip():
                        await event_queue.put(buf if buf.endswith("\n\n") else buf + "\n\n")
            except asyncio.CancelledError:
                raise
            except httpx.TimeoutException as e:
                err = {"type": "error", "error": {"type": "timeout_error", "message": str(e)}}
                await event_queue.put(f"event: error\ndata: {json.dumps(err)}\n\n")
            except Exception as e:
                logger.error(f"[{self.name}] stream error: {e}")
                err = {"type": "error", "error": {"type": "api_error", "message": str(e)}}
                await event_queue.put(f"event: error\ndata: {json.dumps(err)}\n\n")
            finally:
                await event_queue.put(DONE)

        reader_task = asyncio.create_task(_reader())
        ping_interval = settings.stream_ping_interval_sec

        try:
            while True:
                try:
                    item = await asyncio.wait_for(event_queue.get(), timeout=ping_interval)
                except asyncio.TimeoutError:
                    yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"
                    continue
                if item is DONE:
                    break
                yield item
        except (asyncio.CancelledError, GeneratorExit):
            reader_task.cancel()
            try:
                await reader_task
            except BaseException:
                pass
            raise
        finally:
            if not reader_task.done():
                reader_task.cancel()
                try:
                    await reader_task
                except BaseException:
                    pass
