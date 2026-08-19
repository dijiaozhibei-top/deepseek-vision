"""OpenAI Chat Completions HTTP backend.

For upstreams that expose an OpenAI Chat Completions endpoint (DeepSeek's
OpenAI-compatible API, vLLM, aggregators, ...). The backend still speaks the
project's internal Anthropic format to the rest of the proxy — invoke() returns
a MessageResponse and stream() yields Anthropic SSE — so the vision /
web_search / web_fetch middleware work unchanged. Only the wire format to the
upstream differs: POST {base_url}/chat/completions with an OpenAI body, and
OpenAI SSE chunks translated into Anthropic SSE events.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional
from uuid import uuid4

import httpx
from fastapi import HTTPException

from app.backends import LLMBackend
from app.config import settings
from app.schemas import MessageRequest, MessageResponse, Usage

logger = logging.getLogger(__name__)


# --- Request conversion: MessageRequest → OpenAI body ---


def _convert_system(system) -> Optional[str]:
    """Flatten Anthropic system (str or list of text blocks) into one string."""
    if system is None:
        return None
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts = []
        for item in system:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
            elif hasattr(item, "model_dump"):
                d = item.model_dump(exclude_none=True)
                if d.get("type") == "text":
                    parts.append(d.get("text", ""))
            else:
                parts.append(str(item))
        return "\n\n".join(p for p in parts if p) or None
    return str(system) or None


def _blocks_to_oai_content(blocks: List[Any]):
    """Convert a list of Anthropic content blocks to OpenAI content parts.

    Returns a plain string when the result is pure text (smallest payload),
    a list of parts when images are involved, or None when nothing remains.
    """
    text_parts: List[str] = []
    image_parts: List[Any] = []
    for block in blocks:
        b = block if isinstance(block, dict) else (block.model_dump() if hasattr(block, "model_dump") else {})
        bt = b.get("type")
        if bt == "text":
            text_parts.append(b.get("text", ""))
        elif bt == "image":
            source = b.get("source") or {}
            if source.get("url"):
                image_parts.append({"type": "image_url", "image_url": {"url": source["url"]}})
            elif source.get("data"):
                mtype = source.get("media_type") or "image/jpeg"
                image_parts.append(
                    {"type": "image_url", "image_url": {"url": f"data:{mtype};base64,{source['data']}"}}
                )
        elif bt == "document":
            # OpenAI has no portable PDF representation.
            logger.warning("[OpenAI] dropping document content block on the way upstream")
        # thinking / redacted_thinking / compaction / tool_use → skip here
    if image_parts:
        content: List[Any] = []
        if text_parts:
            content.append({"type": "text", "text": "\n\n".join(text_parts)})
        content.extend(image_parts)
        return content
    if text_parts:
        return "\n\n".join(text_parts)
    return None


def _tool_result_content(content) -> Any:
    """Convert an Anthropic tool_result content to an OpenAI tool message content."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: List[str] = []
        image_parts: List[Any] = []
        for block in content:
            b = block if isinstance(block, dict) else (block.model_dump() if hasattr(block, "model_dump") else {})
            bt = b.get("type")
            if bt == "text":
                text_parts.append(b.get("text", ""))
            elif bt == "image":
                source = b.get("source") or {}
                if source.get("url"):
                    image_parts.append({"type": "image_url", "image_url": {"url": source["url"]}})
                elif source.get("data"):
                    mtype = source.get("media_type") or "image/jpeg"
                    image_parts.append(
                        {"type": "image_url", "image_url": {"url": f"data:{mtype};base64,{source['data']}"}}
                    )
            elif bt == "document":
                logger.warning("[OpenAI] dropping document block inside tool_result")
        if image_parts:
            out: List[Any] = []
            if text_parts:
                out.append({"type": "text", "text": "\n\n".join(text_parts)})
            out.extend(image_parts)
            return out
        return "\n\n".join(text_parts)
    return str(content)


def _convert_messages(request: MessageRequest) -> List[Dict[str, Any]]:
    """Flatten Anthropic user/assistant messages into OpenAI messages.

    OpenAI requires tool messages to immediately follow the assistant message
    that produced the tool_calls. The Anthropic invariant — a tool_result always
    lives in the user message right after its assistant turn — makes this a
    lossless expansion.
    """
    out: List[Dict[str, Any]] = []
    sys_text = _convert_system(request.system)
    if sys_text:
        out.append({"role": "system", "content": sys_text})

    for msg in request.messages:
        content = msg.content if isinstance(msg.content, list) else None

        if msg.role == "assistant":
            if content is None:
                out.append({"role": "assistant", "content": msg.content or ""})
                continue
            text_parts: List[str] = []
            reasoning_parts: List[str] = []
            tool_calls: List[Dict[str, Any]] = []
            for block in content:
                b = block if isinstance(block, dict) else (block.model_dump() if hasattr(block, "model_dump") else {})
                bt = b.get("type")
                if bt == "text":
                    text_parts.append(b.get("text", ""))
                elif bt in ("thinking", "redacted_thinking"):
                    # DeepSeek thinking mode requires reasoning to be passed back
                    # verbatim; restore it as reasoning_content on the way out.
                    reasoning_parts.append(b.get("thinking") or b.get("data") or "")
                elif bt == "tool_use":
                    tool_calls.append({
                        "id": b.get("id") or f"call_{uuid4().hex[:16]}",
                        "type": "function",
                        "function": {
                            "name": b.get("name", ""),
                            "arguments": json.dumps(b.get("input", {}), ensure_ascii=False),
                        },
                    })
                # document / compaction → drop
            oai: Dict[str, Any] = {"role": "assistant"}
            if reasoning_parts:
                oai["reasoning_content"] = "\n".join(reasoning_parts)
            if text_parts:
                oai["content"] = "\n\n".join(text_parts)
            if tool_calls:
                oai["tool_calls"] = tool_calls
            if "reasoning_content" not in oai and "content" not in oai and "tool_calls" not in oai:
                continue  # empty turn — drop
            out.append(oai)

        else:  # user
            if content is None:
                out.append({"role": "user", "content": msg.content or ""})
                continue
            tool_results: List[Dict[str, Any]] = []
            rest: List[Any] = []
            for block in content:
                b = block if isinstance(block, dict) else (block.model_dump() if hasattr(block, "model_dump") else {})
                if b.get("type") == "tool_result":
                    tool_results.append(b)
                else:
                    rest.append(b)
            for tr in tool_results:
                out.append({
                    "role": "tool",
                    "tool_call_id": tr.get("tool_use_id", ""),
                    "content": _tool_result_content(tr.get("content")),
                })
            # Non-tool content must come after all tool messages, otherwise it
            # would sit between the assistant tool_calls and the tool results.
            user_content = _blocks_to_oai_content(rest)
            if user_content:
                out.append({"role": "user", "content": user_content})

    return out


def _tools_to_oai(tools: List[Any]) -> List[Dict[str, Any]]:
    result = []
    for tool in tools:
        t = tool if isinstance(tool, dict) else (tool.model_dump() if hasattr(tool, "model_dump") else {})
        result.append({
            "type": "function",
            "function": {
                "name": t.get("name", ""),
                "description": t.get("description") or "",
                "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
            },
        })
    return result


def _tool_choice_to_oai(tool_choice: Any) -> Any:
    if isinstance(tool_choice, str):
        if tool_choice == "any":
            return "required"
        if tool_choice in ("auto", "none"):
            return tool_choice
        return "auto"
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "tool":
        name = tool_choice.get("name")
        if name:
            return {"type": "function", "function": {"name": name}}
    return "auto"


def _build_body(request: MessageRequest, upstream_model_id: str, stream: bool) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "model": upstream_model_id,
        "stream": stream,
        "max_tokens": request.max_tokens,
        "messages": _convert_messages(request),
    }
    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.top_p is not None:
        body["top_p"] = request.top_p
    if request.stop_sequences:
        body["stop"] = list(request.stop_sequences)
    if request.tools:
        body["tools"] = _tools_to_oai(request.tools)
    if request.tool_choice:
        body["tool_choice"] = _tool_choice_to_oai(request.tool_choice)
    if request.metadata and request.metadata.user_id:
        body["user"] = request.metadata.user_id
    if stream:
        body["stream_options"] = {"include_usage": True}

    if upstream_model_id.startswith("deepseek-"):
        # Mirror messages_http: pass through the caller's reasoning effort,
        # defaulting to max so DeepSeek's adaptive thinking stays enabled.
        effort = None
        if isinstance(request.output_config, dict) and isinstance(request.output_config.get("effort"), str):
            effort = request.output_config["effort"] or None
        body["reasoning_effort"] = effort or "max"
    return body


# --- Response conversion: OpenAI JSON → MessageResponse ---

_FINISH_TO_STOP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
}


def _map_finish_reason(finish_reason: Any) -> str:
    if finish_reason == "insufficient_system_resource":
        logger.warning("[OpenAI] upstream returned insufficient_system_resource")
    return _FINISH_TO_STOP.get(finish_reason, "end_turn")


def _map_usage(usage_data: Dict[str, Any]) -> Usage:
    prompt = usage_data.get("prompt_tokens") or 0
    completion = usage_data.get("completion_tokens") or 0
    details = usage_data.get("prompt_tokens_details") or {}
    # OpenAI reports cached tokens inside prompt_tokens_details; DeepSeek exposes
    # top-level prompt_cache_hit_tokens / prompt_cache_miss_tokens. cache_creation
    # here is an approximation (missed ≠ freshly written).
    cache_hit = details.get("cached_tokens") or usage_data.get("prompt_cache_hit_tokens")
    cache_miss = usage_data.get("prompt_cache_miss_tokens")
    return Usage(
        input_tokens=prompt,
        output_tokens=completion,
        cache_read_input_tokens=cache_hit if cache_hit is not None else None,
        cache_creation_input_tokens=cache_miss if cache_miss is not None else None,
    )


def _parse_response(response_body: Dict[str, Any], client_model: str, message_id: str) -> MessageResponse:
    choice = (response_body.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = msg.get("content")
    blocks: List[Dict[str, Any]] = []
    # DeepSeek's OpenAI endpoint reports reasoning in reasoning_content (sibling
    # of content); surface it as an Anthropic thinking block, mirroring the
    # Messages-format upstream behavior.
    reasoning = msg.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        blocks.append({"type": "thinking", "thinking": reasoning})
    if isinstance(content, str) and content:
        blocks.append({"type": "text", "text": content})
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                blocks.append({"type": "text", "text": part.get("text", "")})

    for tc in (msg.get("tool_calls") or []):
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except Exception:
            logger.warning("[OpenAI] failed to parse tool_call arguments, using {}")
            args = {}
        blocks.append({
            "type": "tool_use",
            "id": tc.get("id") or f"toolu_{uuid4().hex[:16]}",
            "name": fn.get("name", ""),
            "input": args,
        })

    return MessageResponse(
        id=message_id,
        content=blocks,
        model=client_model,
        stop_reason=_map_finish_reason(choice.get("finish_reason")),
        usage=_map_usage(response_body.get("usage") or {}),
    )


def _raise_for_http_error(resp: httpx.Response, backend_name: str) -> None:
    try:
        body = resp.json()
    except ValueError:
        body = None
    err = (body or {}).get("error") if isinstance(body, dict) else None
    if not isinstance(err, dict):
        err = {"type": "api_error", "message": str(body)[:500]}
    logger.error(f"[{backend_name}] upstream {resp.status_code}: {err}")
    raise HTTPException(
        status_code=resp.status_code,
        detail={"type": err.get("type", "api_error"), "message": err.get("message", "upstream error")},
    )


# --- Streaming: OpenAI SSE chunks → Anthropic SSE events ---


def _sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


class _StreamConverter:
    """Translates OpenAI streaming deltas into Anthropic SSE strings.

    OpenAI gives text (delta.content), reasoning (delta.reasoning_content) and
    tool_calls as separate delta fields with no explicit per-block boundaries, so
    content blocks are opened lazily in first-seen order and closed when the next
    phase begins or at finalize. text/thinking blocks are emitted incrementally;
    tool_calls aggregate partial_json per call and are closed at finalize.
    """

    def __init__(self, client_model: str, request_id: str) -> None:
        self.client_model = client_model
        self.request_id = request_id
        self.next_index = 0
        self.opened: List[tuple] = []            # (anthropic_index, kind)
        self._tool_ai_index: Dict[int, int] = {}  # openai tool index → anthropic index
        self._tool_states: Dict[int, Dict[str, Any]] = {}
        self._late_text: List[str] = []
        self._text_opened = False
        self._text_index: Optional[int] = None
        self._thinking_opened = False
        self._thinking_index: Optional[int] = None
        self._tool_phase = False
        self._message_started = False
        self._finalized = False
        self._finish_reason: Optional[str] = None
        self._usage: Dict[str, Any] = {}

    # --- event builders ---

    def _message_start(self) -> str:
        return _sse("message_start", {
            "type": "message_start",
            "message": {
                "id": self.request_id,
                "type": "message",
                "role": "assistant",
                "model": self.client_model,
                "content": [],
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        })

    def _start_block(self, index: int, content_block: Dict[str, Any]) -> str:
        return _sse("content_block_start", {"type": "content_block_start", "index": index, "content_block": content_block})

    def _stop_block(self, index: int) -> str:
        return _sse("content_block_stop", {"type": "content_block_stop", "index": index})

    def _delta(self, index: int, delta: Dict[str, Any]) -> str:
        return _sse("content_block_delta", {"type": "content_block_delta", "index": index, "delta": delta})

    # --- public API ---

    def on_chunk(self, obj: Dict[str, Any]) -> List[str]:
        out: List[str] = []
        choices = obj.get("choices") or []
        if not choices:
            usage = obj.get("usage")
            if usage:
                self._usage = usage
            return out
        choice = choices[0]
        if choice.get("finish_reason"):
            self._finish_reason = choice.get("finish_reason")
        delta = choice.get("delta") or {}
        if delta.get("reasoning_content"):
            out.extend(self._on_thinking(delta["reasoning_content"]))
        if delta.get("content"):
            out.extend(self._on_text(delta["content"]))
        if delta.get("tool_calls"):
            out.extend(self._on_tool_calls(delta["tool_calls"]))
        return out

    def finalize(self) -> List[str]:
        if self._finalized:
            return []
        self._finalized = True
        out: List[str] = []
        if not self._message_started:
            out.append(self._message_start())
            self._message_started = True
        while self.opened:
            index, _ = self.opened.pop()
            out.append(self._stop_block(index))
        if self._late_text:
            index = self.next_index
            text = "".join(self._late_text)
            out.append(self._start_block(index, {"type": "text", "text": ""}))
            out.append(self._delta(index, {"type": "text_delta", "text": text}))
            out.append(self._stop_block(index))
            self.next_index += 1

        usage: Dict[str, Any] = {"output_tokens": self._usage.get("completion_tokens") or 0}
        cache_hit = None
        details = self._usage.get("prompt_tokens_details") or {}
        cache_hit = details.get("cached_tokens") or self._usage.get("prompt_cache_hit_tokens")
        if cache_hit is not None:
            usage["cache_read_input_tokens"] = cache_hit
        cache_miss = self._usage.get("prompt_cache_miss_tokens")
        if cache_miss is not None:
            usage["cache_creation_input_tokens"] = cache_miss

        out.append(_sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": _map_finish_reason(self._finish_reason)},
            "usage": usage,
        }))
        out.append(_sse("message_stop", {"type": "message_stop"}))
        return out

    # --- per-field handlers ---

    def _on_thinking(self, fragment: str) -> List[str]:
        out: List[str] = []
        if not self._message_started:
            out.append(self._message_start())
            self._message_started = True
        if self._text_opened or self._tool_phase:
            return out  # late reasoning has no valid slot — drop
        if not self._thinking_opened:
            index = self.next_index
            self._thinking_index = index
            out.append(self._start_block(index, {"type": "thinking", "thinking": ""}))
            self._thinking_opened = True
            self.opened.append((index, "thinking"))
            self.next_index += 1
        else:
            index = self._thinking_index
        out.append(self._delta(index, {"type": "thinking_delta", "thinking": fragment}))
        return out

    def _on_text(self, fragment: str) -> List[str]:
        out: List[str] = []
        if not self._message_started:
            out.append(self._message_start())
            self._message_started = True
        if self._tool_phase:
            self._late_text.append(fragment)
            return out
        if not self._text_opened:
            while self.opened:
                index, kind = self.opened.pop()
                out.append(self._stop_block(index))
                if kind == "thinking":
                    break
            index = self.next_index
            self._text_index = index
            out.append(self._start_block(index, {"type": "text", "text": ""}))
            self._text_opened = True
            self.opened.append((index, "text"))
            self.next_index += 1
        else:
            index = self._text_index
        out.append(self._delta(index, {"type": "text_delta", "text": fragment}))
        return out

    def _on_tool_calls(self, tool_calls: List[Any]) -> List[str]:
        out: List[str] = []
        if not self._message_started:
            out.append(self._message_start())
            self._message_started = True
        if not self._tool_phase:
            while self.opened:
                index, _ = self.opened.pop()
                out.append(self._stop_block(index))
            self._tool_phase = True
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            oai_index = tc.get("index")
            if oai_index is None:
                continue
            ai_index = self._tool_ai_index.get(oai_index)
            if ai_index is None:
                fn = tc.get("function") or {}
                tool_id = tc.get("id") or f"toolu_{uuid4().hex[:16]}"
                name = fn.get("name") or ""
                ai_index = self.next_index
                self._tool_ai_index[oai_index] = ai_index
                self._tool_states[oai_index] = {"id": tool_id, "name": name}
                out.append(self._start_block(ai_index, {"type": "tool_use", "id": tool_id, "name": name, "input": {}}))
                self.opened.append((ai_index, "tool"))
                self.next_index += 1
            else:
                state = self._tool_states.get(oai_index) or {}
                fn = tc.get("function") or {}
                tc_id = tc.get("id")
                if tc_id and tc_id != state.get("id"):
                    state["id"] = tc_id
                if fn.get("name"):
                    state["name"] = fn["name"]
            args_frag = (tc.get("function") or {}).get("arguments")
            if args_frag:
                out.append(self._delta(ai_index, {"type": "input_json_delta", "partial_json": args_frag}))
        return out


class OpenAICompletionsBackend(LLMBackend):
    """OpenAI Chat Completions-compatible upstream backend."""

    def __init__(self, name: str, base_url: str, api_key: str, model_map: Dict[str, str]) -> None:
        self.name = name
        self.model_map = dict(model_map)
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(float(settings.upstream_timeout), connect=10.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=50),
        )
        self._stream_client = httpx.AsyncClient(
            timeout=httpx.Timeout(float(settings.upstream_stream_timeout), connect=10.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=50),
        )

    def _headers(self, stream: bool) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "content-type": "application/json",
            "accept": "text/event-stream" if stream else "application/json",
        }

    async def invoke(
        self,
        request: MessageRequest,
        request_id: str,
        anthropic_beta: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> MessageResponse:
        upstream_model_id = self.resolve_model_id(request.model)
        body = _build_body(request, upstream_model_id, stream=False)
        headers = self._headers(stream=False)

        if meta is not None:
            meta["model_id"] = upstream_model_id
            meta["request_size_bytes"] = len(json.dumps(body).encode("utf-8"))

        if settings.debug_upstream:
            logger.info(f"[{self.name}] Request body: {json.dumps(body, ensure_ascii=False)[:2000]}")

        try:
            resp = await self._client.post(f"{self._base_url}/chat/completions", headers=headers, json=body)
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
        headers = self._headers(stream=True)

        if meta is not None:
            meta["model_id"] = upstream_model_id

        yield _sse("ping", {"type": "ping"})

        event_queue: asyncio.Queue = asyncio.Queue(maxsize=1024)
        DONE = object()
        converter = _StreamConverter(request.model, request_id)

        async def _reader():
            try:
                async with self._stream_client.stream(
                    "POST",
                    f"{self._base_url}/chat/completions",
                    headers=headers,
                    json=body,
                ) as resp:
                    if resp.status_code >= 400:
                        await resp.aread()
                        try:
                            _raise_for_http_error(resp, self.name)
                        except HTTPException as http_exc:
                            detail = http_exc.detail if isinstance(http_exc.detail, dict) else {"type": "api_error", "message": str(http_exc.detail)}
                            await event_queue.put(_sse("error", {"type": "error", "error": detail}))
                        return

                    buf = ""
                    async for chunk in resp.aiter_text():
                        if not chunk:
                            continue
                        buf += chunk
                        while "\n" in buf:
                            line, buf = buf.split("\n", 1)
                            line = line.strip()
                            if not line.startswith("data:"):
                                continue
                            payload = line[5:].strip()
                            if payload == "[DONE]":
                                for s in converter.finalize():
                                    await event_queue.put(s)
                                return
                            try:
                                obj = json.loads(payload)
                            except Exception:
                                logger.warning(f"[{self.name}] skipped unparseable stream chunk")
                                continue
                            for s in converter.on_chunk(obj):
                                await event_queue.put(s)
                    # Upstream ended without [DONE] — finalize anyway.
                    for s in converter.finalize():
                        await event_queue.put(s)
            except asyncio.CancelledError:
                raise
            except httpx.TimeoutException as e:
                err = {"type": "error", "error": {"type": "timeout_error", "message": str(e)}}
                await event_queue.put(_sse("error", err))
            except Exception as e:
                logger.error(f"[{self.name}] stream error: {e}")
                err = {"type": "error", "error": {"type": "api_error", "message": str(e)}}
                await event_queue.put(_sse("error", err))
            finally:
                await event_queue.put(DONE)

        reader_task = asyncio.create_task(_reader())
        ping_interval = settings.stream_ping_interval_sec

        try:
            while True:
                try:
                    item = await asyncio.wait_for(event_queue.get(), timeout=ping_interval)
                except asyncio.TimeoutError:
                    yield _sse("ping", {"type": "ping"})
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
