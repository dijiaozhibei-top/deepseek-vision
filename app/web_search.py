"""
Web Search — two-round architecture.

Round 1: LLM emits search queries → parallel execution via asyncio.gather.
Round 2: LLM generates final answer from all results (no further searching).

This avoids the O(n²) token growth of multi-iteration loops.
"""
import asyncio
import base64
import json
import logging
import re
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from app.backends import LLMBackend
from app.config import settings
from app.schemas import MessageRequest, MessageResponse, Usage

logger = logging.getLogger(__name__)

WEB_SEARCH_TOOL_TYPES = {"web_search_20250305", "web_search_20260209"}
WEB_SEARCH_BETA_HEADERS = {"web-search-2025-03-05", "web-search-2026-02-09"}

_CITATION_MARKER_RE = re.compile(r"\[(\d+)\]")

_CITATION_SYSTEM_PROMPT = (
    "When you use web search results to answer questions, you MUST cite sources "
    "using numbered references in square brackets. The search results are numbered "
    "[Result 1], [Result 2], etc. After each factual claim based on a search result, "
    "append the result number like this: 'Python 3.13 was released in October 2024 [1].' "
    "Multiple sources can be combined: 'This is widely used [1][3].' "
    "Every claim from search results MUST have at least one [N] citation. "
    "Do NOT omit citations."
)

_CITATION_REMINDER = (
    "\n\n[Remember: cite every claim from these results using [N] notation, "
    "where N is the Result number shown above.]"
)

_SEARCH_PLANNING_PROMPT = (
    "When you need to search the web, emit ALL your search queries at once as separate "
    "web_search tool calls in this single response. Use 2-4 complementary queries that "
    "cover different aspects or phrasings of the question. You will only get one chance "
    "to search — make your queries comprehensive and diverse.\n"
    "Example: For 'What are the latest developments in quantum computing?', emit:\n"
    "- web_search('quantum computing breakthroughs 2026')\n"
    "- web_search('quantum computing industry news latest')\n"
    "- web_search('quantum error correction recent advances')\n"
    "Do NOT emit just one generic query when the topic benefits from multiple angles."
)


def encode_content(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def decode_content(encoded: str) -> str:
    return base64.b64decode(encoded.encode("ascii")).decode("utf-8")


# --- Search providers ---

def _get_tavily_url() -> str:
    return "https://api.tavily.com/search"


def _get_search_provider():
    provider = settings.web_search_provider
    api_key = settings.tavily_api_key if provider == "tavily" else settings.brave_api_key
    if not api_key:
        raise ValueError(f"Web search API key is required. Set TAVILY_API_KEY or BRAVE_API_KEY.")
    if provider not in ("tavily", "brave"):
        raise ValueError(f"Unknown search provider: {provider}")
    return provider, api_key


async def _execute_search(
    query: str,
    max_results: int,
    allowed_domains: Optional[List[str]] = None,
    blocked_domains: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """Execute search and return list of {url, title, content, page_age?}."""
    provider_type, api_key = _get_search_provider()

    if provider_type == "tavily":
        payload: Dict[str, Any] = {
            "api_key": api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
        }
        if allowed_domains:
            payload["include_domains"] = allowed_domains
        if blocked_domains:
            payload["exclude_domains"] = blocked_domains
        async with httpx.AsyncClient(timeout=30.0) as http:
            resp = await http.post(_get_tavily_url(), json=payload)
            resp.raise_for_status()
            response = resp.json()
        return [
            {
                "url": item.get("url", ""),
                "title": item.get("title", ""),
                "content": item.get("content", ""),
            }
            for item in response.get("results", [])
        ]

    elif provider_type == "brave":
        search_query = query
        if allowed_domains:
            site_filter = " OR ".join(f"site:{d}" for d in allowed_domains)
            search_query = f"({site_filter}) {query}"
        params = {"q": search_query, "count": max_results}
        headers = {"Accept": "application/json", "X-Subscription-Token": api_key}
        async with httpx.AsyncClient(timeout=15.0) as http:
            resp = await http.get("https://api.search.brave.com/res/v1/web/search", params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        return [
            {
                "url": item.get("url", ""),
                "title": item.get("title", ""),
                "content": item.get("description", ""),
            }
            for item in data.get("web", {}).get("results", [])[:max_results]
        ]

    return []


# --- Tool definitions ---

_WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": (
        "Search the web for current information. Returns results with URLs, "
        "titles, and content snippets. Always cite your sources when using results."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "The search query"}},
        "required": ["query"],
    },
}


def _build_tools(original_tools: Optional[List[Any]]) -> List[Any]:
    """Replace web_search marker with custom tool definition."""
    if not original_tools:
        return [_WEB_SEARCH_TOOL]
    result = []
    for tool in original_tools:
        t = tool if isinstance(tool, dict) else (tool.model_dump() if hasattr(tool, "model_dump") else {})
        if t.get("type", "") in WEB_SEARCH_TOOL_TYPES:
            continue
        result.append(t)
    result.append(_WEB_SEARCH_TOOL)
    return result


def _build_tools_without_search(original_tools: Optional[List[Any]]) -> Optional[List[Any]]:
    """Return non-search tools only (for the answer-generation round)."""
    if not original_tools:
        return None
    result = []
    for tool in original_tools:
        t = tool if isinstance(tool, dict) else (tool.model_dump() if hasattr(tool, "model_dump") else {})
        if t.get("type", "") in WEB_SEARCH_TOOL_TYPES:
            continue
        if t.get("name") == "web_search":
            continue
        result.append(t)
    return result or None


def _extract_config(request: MessageRequest) -> Dict[str, Any]:
    """Extract web search config (max_uses, allowed_domains, blocked_domains)."""
    config = {"max_uses": settings.web_search_default_max_uses}
    if not request.tools:
        return config
    for tool in request.tools:
        t = tool if isinstance(tool, dict) else (tool.model_dump() if hasattr(tool, "model_dump") else {})
        if t.get("type", "") in WEB_SEARCH_TOOL_TYPES:
            config["max_uses"] = min(t.get("max_uses") or settings.web_search_default_max_uses, settings.web_search_default_max_uses)
            config["allowed_domains"] = t.get("allowed_domains")
            config["blocked_domains"] = t.get("blocked_domains")
            break
    return config


def _filter_beta(anthropic_beta: Optional[str]) -> Optional[str]:
    if not anthropic_beta:
        return None
    filtered = [h.strip() for h in anthropic_beta.split(",") if h.strip() not in WEB_SEARCH_BETA_HEADERS]
    return ",".join(filtered) or None


def _inject_citation_system(system) -> List[Dict[str, Any]]:
    """Inject citation instruction into system prompt."""
    citation_block = {"type": "text", "text": _CITATION_SYSTEM_PROMPT}
    if system is None:
        return [citation_block]
    if isinstance(system, str):
        return [{"type": "text", "text": system}, citation_block]
    if isinstance(system, list):
        augmented = []
        for item in system:
            if hasattr(item, "model_dump"):
                augmented.append(item.model_dump(exclude_none=True))
            elif isinstance(item, dict):
                augmented.append(item)
            else:
                augmented.append({"type": "text", "text": str(item)})
        augmented.append(citation_block)
        return augmented
    return [{"type": "text", "text": str(system)}, citation_block]


def _inject_search_planning_system(system) -> List[Dict[str, Any]]:
    """Inject search planning instruction into system prompt (Round 1)."""
    planning_block = {"type": "text", "text": _SEARCH_PLANNING_PROMPT}
    if system is None:
        return [planning_block]
    if isinstance(system, str):
        return [{"type": "text", "text": system}, planning_block]
    if isinstance(system, list):
        augmented = []
        for item in system:
            if hasattr(item, "model_dump"):
                augmented.append(item.model_dump(exclude_none=True))
            elif isinstance(item, dict):
                augmented.append(item)
            else:
                augmented.append({"type": "text", "text": str(item)})
        augmented.append(planning_block)
        return augmented
    return [{"type": "text", "text": str(system)}, planning_block]


def _to_server_id(original_id: str) -> str:
    if original_id.startswith("srvtoolu_"):
        return original_id
    if original_id.startswith("toolu_"):
        return "srvtoolu_" + original_id[6:]
    return f"srvtoolu_{original_id}"


def _convert_to_server_tool_use(content: list) -> list:
    """Convert web_search tool_use blocks to server_tool_use."""
    converted = []
    for block in content:
        bd = block if isinstance(block, dict) else (block.model_dump() if hasattr(block, "model_dump") else {})
        if bd.get("type") == "tool_use" and bd.get("name") == "web_search":
            converted.append({
                "type": "server_tool_use",
                "id": _to_server_id(bd.get("id", "")),
                "name": "web_search",
                "input": bd.get("input", {}),
            })
        else:
            converted.append(bd)
    return converted


def _build_web_search_result(tool_use_id: str, results: List[Dict]) -> Dict[str, Any]:
    search_results = []
    for r in results:
        entry = {
            "type": "web_search_result",
            "url": r["url"],
            "title": r["title"],
            "encrypted_content": encode_content(r.get("content", "")),
        }
        search_results.append(entry)
    return {"type": "web_search_tool_result", "tool_use_id": tool_use_id, "content": search_results}


def _build_web_search_error(tool_use_id: str, error_code: str) -> Dict[str, Any]:
    return {
        "type": "web_search_tool_result",
        "tool_use_id": tool_use_id,
        "content": {"type": "web_search_tool_result_error", "error_code": error_code},
    }


def _build_continuation_messages(
    messages: list,
    response_content: list,
    tool_results: list,
    result_registry: Dict[int, Dict[str, str]],
) -> list:
    """Build messages for next iteration with numbered results."""
    new_messages = list(messages)

    # Assistant message with original content
    assistant_content = []
    for block in response_content:
        bd = block if isinstance(block, dict) else (block.model_dump() if hasattr(block, "model_dump") else {})
        assistant_content.append(bd)
    new_messages.append({"role": "assistant", "content": assistant_content})

    # User message with tool_result blocks
    user_content = []
    for result in tool_results:
        tool_use_id = result.get("tool_use_id", "")
        result_content = result.get("content", {})

        if result.get("type") == "web_search_tool_result" and isinstance(result_content, list):
            text_parts = []
            for sr in result_content:
                enc = sr.get("encrypted_content", "")
                try:
                    content = decode_content(enc) if enc else ""
                except Exception:
                    content = enc
                idx = len(result_registry) + 1
                result_registry[idx] = {
                    "url": sr.get("url", ""),
                    "title": sr.get("title", ""),
                    "content": content,
                    "encrypted_index": encode_content(str(idx)),
                }
                text_parts.append(f"[Result {idx}]\nTitle: {sr.get('title', '')}\nURL: {sr.get('url', '')}\nContent: {content}")
            result_text = "\n\n---\n\n".join(text_parts) + _CITATION_REMINDER
            user_content.append({"type": "tool_result", "tool_use_id": tool_use_id, "content": result_text})
        elif isinstance(result_content, dict):
            user_content.append({
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": f"Error: {result_content.get('error_code', 'unknown')}",
                "is_error": True,
            })
        else:
            user_content.append({"type": "tool_result", "tool_use_id": tool_use_id, "content": str(result_content)})

    new_messages.append({"role": "user", "content": user_content})
    return new_messages


def _post_process_citations(content_blocks: list, result_registry: Dict[int, Dict[str, str]]) -> list:
    """Convert [N] markers in text blocks to citation objects."""
    if not result_registry:
        return content_blocks

    processed = []
    for block in content_blocks:
        bd = block if isinstance(block, dict) else (block.model_dump() if hasattr(block, "model_dump") else {})
        if bd.get("type") != "text":
            processed.append(bd)
            continue

        text = bd.get("text", "")
        if not text or not _CITATION_MARKER_RE.search(text):
            processed.append(bd)
            continue

        segments = []
        last_end = 0
        for match in re.finditer(r"((?:\[\d+\])+)", text):
            marker_start = match.start()
            marker_end = match.end()
            cited_indices = [int(m) for m in re.findall(r"\[(\d+)\]", match.group(0))]
            cited_segment = text[last_end:marker_start]
            last_end = marker_end

            if not cited_segment.strip():
                continue

            citations = []
            for idx in cited_indices:
                info = result_registry.get(idx)
                if not info:
                    continue
                citations.append({
                    "type": "web_search_result_location",
                    "url": info.get("url", ""),
                    "title": info.get("title", ""),
                    "encrypted_index": info.get("encrypted_index", ""),
                    "cited_text": info.get("content", "")[:150],
                })

            if citations:
                segments.append({"type": "text", "text": cited_segment.rstrip(), "citations": citations})
            else:
                segments.append({"type": "text", "text": cited_segment + match.group(0)})

        remaining = text[last_end:].strip()
        if remaining:
            segments.append({"type": "text", "text": remaining})
        processed.extend(segments if segments else [bd])

    return processed


# --- Main handlers ---

async def handle_web_search(
    request: MessageRequest,
    request_id: str,
    anthropic_beta: Optional[str] = None,
    backend: Optional[LLMBackend] = None,
) -> MessageResponse:
    """Two-round web search: plan queries → parallel fetch → generate answer."""
    if backend is None:
        raise ValueError("backend must be provided")

    config = _extract_config(request)
    max_uses = config["max_uses"]
    filtered_beta = _filter_beta(anthropic_beta)
    result_registry: Dict[int, Dict[str, str]] = {}
    all_content: List[Any] = []
    total_input = total_output = 0
    total_cache_creation = total_cache_read = 0
    messages: list = [
        m.model_dump(exclude_none=True) if hasattr(m, "model_dump") else m
        for m in request.messages
    ]

    # === Round 1: Let the model decide what to search ===
    logger.info(f"[WebSearch] Round 1: planning queries (max_uses={max_uses})")
    plan_request = MessageRequest(
        model=request.model,
        messages=messages,
        max_tokens=request.max_tokens,
        system=_inject_search_planning_system(request.system),
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stop_sequences=request.stop_sequences,
        stream=False,
        tools=_build_tools(request.tools),
        tool_choice=request.tool_choice,
        thinking=request.thinking,
        metadata=request.metadata,
        output_config=request.output_config,
        context_management=request.context_management,
    )

    plan_response = await backend.invoke(plan_request, f"{request_id}_plan", filtered_beta)
    total_input += plan_response.usage.input_tokens
    total_output += plan_response.usage.output_tokens
    total_cache_creation += plan_response.usage.cache_creation_input_tokens or 0
    total_cache_read += plan_response.usage.cache_read_input_tokens or 0
    response_content = [
        b if isinstance(b, dict) else (b.model_dump() if hasattr(b, "model_dump") else b)
        for b in (plan_response.content or [])
    ]

    web_search_uses = [
        b for b in response_content
        if b.get("type") == "tool_use" and b.get("name") == "web_search"
    ]

    if not web_search_uses or plan_response.stop_reason != "tool_use":
        all_content.extend(_convert_to_server_tool_use(response_content))
        all_content = _post_process_citations(all_content, result_registry)
        usage = Usage(
            input_tokens=total_input,
            output_tokens=total_output,
            cache_creation_input_tokens=total_cache_creation or None,
            cache_read_input_tokens=total_cache_read or None,
        )
        return MessageResponse(
            id=request_id, content=all_content, model=request.model,
            stop_reason=plan_response.stop_reason, usage=usage,
        )

    # === Execute searches in parallel ===
    queries_to_run = web_search_uses[:max_uses]
    logger.info(f"[WebSearch] Executing {len(queries_to_run)} searches in parallel")

    search_tasks = []
    for tool_use in queries_to_run:
        query = (tool_use.get("input") or {}).get("query", "").strip()
        if not query:
            logger.warning(f"[WebSearch] Skipping empty query from tool_use {tool_use.get('id')}")

            async def _empty_result():
                return []

            search_tasks.append(_empty_result())
        else:
            search_tasks.append(
                _execute_search(
                    query, settings.web_search_max_results,
                    config.get("allowed_domains"), config.get("blocked_domains"),
                )
            )

    raw_results = await asyncio.gather(*search_tasks, return_exceptions=True)

    # Build results, handling failures gracefully
    search_count = 0
    continuation_results: List[Dict[str, Any]] = []

    for i, (tool_use, result) in enumerate(zip(queries_to_run, raw_results)):
        original_id = tool_use.get("id", "")
        server_id = _to_server_id(original_id)

        if isinstance(result, Exception):
            logger.error(f"[WebSearch] Search {i+1} failed: {result}")
            client_result = _build_web_search_error(server_id, "unavailable")
            cont_result = _build_web_search_error(original_id, "unavailable")
        else:
            client_result = _build_web_search_result(server_id, result)
            cont_result = _build_web_search_result(original_id, result)
            search_count += 1

        continuation_results.append(cont_result)
        all_content.append({"type": "server_tool_use", "id": server_id, "name": "web_search", "input": tool_use.get("input", {})})
        all_content.append(client_result)

    # Mark excess tool_uses (beyond max_uses) as errors for the client
    for tool_use in web_search_uses[max_uses:]:
        server_id = _to_server_id(tool_use.get("id", ""))
        original_id = tool_use.get("id", "")
        all_content.append({"type": "server_tool_use", "id": server_id, "name": "web_search", "input": tool_use.get("input", {})})
        all_content.append(_build_web_search_error(server_id, "max_uses_exceeded"))
        continuation_results.append(_build_web_search_error(original_id, "max_uses_exceeded"))

    logger.info(f"[WebSearch] {search_count}/{len(queries_to_run)} searches succeeded")

    # Build continuation messages for Round 2
    messages_with_results = _build_continuation_messages(
        messages, response_content, continuation_results, result_registry,
    )

    # === Round 2: Generate final answer from search results ===
    logger.info("[WebSearch] Round 2: generating answer from search results")
    answer_request = MessageRequest(
        model=request.model,
        messages=messages_with_results,
        max_tokens=request.max_tokens,
        system=_inject_citation_system(request.system),
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stop_sequences=request.stop_sequences,
        stream=False,
        tools=_build_tools_without_search(request.tools),
        tool_choice=None,
        thinking=request.thinking,
        metadata=request.metadata,
        output_config=request.output_config,
        context_management=request.context_management,
    )

    answer_response = await backend.invoke(answer_request, f"{request_id}_answer", filtered_beta)
    total_input += answer_response.usage.input_tokens
    total_output += answer_response.usage.output_tokens
    total_cache_creation += answer_response.usage.cache_creation_input_tokens or 0
    total_cache_read += answer_response.usage.cache_read_input_tokens or 0

    answer_content = answer_response.content or []
    for block in answer_content:
        bd = block if isinstance(block, dict) else (block.model_dump(exclude_none=True) if hasattr(block, "model_dump") else block)
        all_content.append(bd)

    # Post-process citations
    all_content = _post_process_citations(all_content, result_registry)

    usage = Usage(
        input_tokens=total_input,
        output_tokens=total_output,
        cache_creation_input_tokens=total_cache_creation or None,
        cache_read_input_tokens=total_cache_read or None,
        server_tool_use={"web_search_requests": search_count} if search_count > 0 else None,
    )

    return MessageResponse(
        id=request_id,
        content=all_content,
        model=request.model,
        stop_reason=answer_response.stop_reason,
        usage=usage,
    )


async def stream_web_search(
    request: MessageRequest,
    request_id: str,
    anthropic_beta: Optional[str] = None,
    backend: Optional[LLMBackend] = None,
) -> AsyncGenerator[str, None]:
    """Streaming web search — hybrid approach (non-streaming backend, SSE per iteration)."""
    # Emit an immediate "connect" ping to prime middlebox buffers and signal liveness.
    yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"
    # Run the underlying (non-streaming) work as a task and emit keep-alive pings
    # while we wait, so clients / CloudFront don't time out on long runs.
    task = asyncio.create_task(handle_web_search(request, request_id, anthropic_beta, backend))
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=settings.stream_ping_interval_sec)
            if done:
                break
            yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"
    except (asyncio.CancelledError, GeneratorExit):
        # Client disconnected — cancel the underlying work to free resources.
        task.cancel()
        try:
            await task
        except BaseException:
            pass
        raise

    exc = task.exception()
    if exc is not None:
        error = {"type": "error", "error": {"type": "api_error", "message": str(exc)}}
        yield f"event: error\ndata: {json.dumps(error)}\n\n"
        return
    response = task.result()

    # Emit message_start
    msg_start = {
        "type": "message_start",
        "message": {
            "id": response.id,
            "type": "message",
            "role": "assistant",
            "model": response.model,
            "content": [],
            "usage": {"input_tokens": response.usage.input_tokens, "output_tokens": 0},
        },
    }
    yield f"event: message_start\ndata: {json.dumps(msg_start)}\n\n"

    # Emit content blocks
    for idx, block in enumerate(response.content):
        bd = block if isinstance(block, dict) else (block.model_dump(exclude_none=True) if hasattr(block, "model_dump") else block)
        block_type = bd.get("type", "text")

        # content_block_start
        if block_type == "text":
            start_block = {"type": "text", "text": ""}
            if bd.get("citations"):
                start_block["citations"] = []
        elif block_type in ("server_tool_use", "web_search_tool_result"):
            start_block = bd
        else:
            start_block = bd

        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': idx, 'content_block': start_block})}\n\n"

        # content_block_delta for text
        if block_type == "text":
            delta = {"type": "text_delta", "text": bd.get("text", "")}
            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': idx, 'delta': delta})}\n\n"
            for citation in bd.get("citations") or []:
                citation_delta = {"type": "citations_delta", "citation": citation}
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': idx, 'delta': citation_delta})}\n\n"

        # content_block_stop
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': idx})}\n\n"

    # message_delta + message_stop
    delta_usage: Dict[str, Any] = {"output_tokens": response.usage.output_tokens}
    if response.usage.server_tool_use:
        delta_usage["server_tool_use"] = response.usage.server_tool_use
    msg_delta = {
        "type": "message_delta",
        "delta": {"stop_reason": response.stop_reason},
        "usage": delta_usage,
    }
    yield f"event: message_delta\ndata: {json.dumps(msg_delta)}\n\n"
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
