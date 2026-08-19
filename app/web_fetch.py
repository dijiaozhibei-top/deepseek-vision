"""
Web Fetch agentic loop.

Intercepts web_fetch tool calls, executes fetches via httpx with SSRF protection,
feeds results back to Claude in a loop, then post-processes [N] citation markers.
"""
import asyncio
import html as html_module
import ipaddress
import json
import logging
import re
import socket
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional
from urllib.parse import urlparse
from uuid import uuid4

import httpcore
import httpx

from app.backends import LLMBackend
from app.config import settings
from app.schemas import MessageRequest, MessageResponse, Usage

logger = logging.getLogger(__name__)

WEB_FETCH_TOOL_TYPES = {"web_fetch_20250910", "web_fetch_20260209"}
WEB_FETCH_BETA_HEADERS = {"web-fetch-2025-09-10", "web-fetch-2026-02-09"}
MAX_ITERATIONS = 25

_CITATION_MARKER_RE = re.compile(r"\[(\d+)\]")

_CITATION_SYSTEM_PROMPT = (
    "When you use content from fetched web pages to answer questions, you MUST cite sources "
    "using numbered references in square brackets. The fetched documents are numbered "
    "[Document 1], [Document 2], etc. After each factual claim based on a fetched document, "
    "append the document number like this: 'Python 3.13 was released in October 2024 [1].' "
    "Multiple sources can be combined: 'This is widely used [1][3].' "
    "Every claim from fetched documents MUST have at least one [N] citation. "
    "Do NOT omit citations."
)

_CITATION_REMINDER = (
    "\n\n[Remember: cite every claim from these fetched documents using [N] notation, "
    "where N is the Document number shown above.]"
)


# --- SSRF Protection ---

def _is_private_ip(ip_str: str) -> bool:
    """Check if an IP address is private, reserved, loopback, or link-local."""
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return True
    return (
        addr.is_private
        or addr.is_reserved
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_unspecified
        or ip_str == "169.254.169.254"   # AWS EC2 metadata
        or ip_str == "169.254.170.2"     # ECS metadata
    )


def _validate_url_ssrf(url: str) -> None:
    """Validate URL against SSRF attacks."""
    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        raise _FetchError("invalid_input", f"Cannot extract hostname from URL: {url}")

    blocked_hostnames = {"localhost", "metadata.google.internal", "metadata.google"}
    if hostname.lower() in blocked_hostnames:
        raise _FetchError("ssrf_blocked", f"Access to internal host is not allowed: {hostname}")

    try:
        addrinfos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror:
        raise _FetchError("url_not_accessible", f"Cannot resolve hostname: {hostname}")

    for addrinfo in addrinfos:
        ip_str = str(addrinfo[4][0])
        if _is_private_ip(ip_str):
            raise _FetchError("ssrf_blocked", f"Access to private/internal IP is not allowed: {hostname} -> {ip_str}")


def _resolve_and_validate(hostname: str) -> str:
    """Resolve hostname, validate ALL IPs against SSRF, return first valid IP for DNS pinning."""
    try:
        addrinfos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror:
        raise _FetchError("url_not_accessible", f"Cannot resolve hostname: {hostname}")
    if not addrinfos:
        raise _FetchError("url_not_accessible", f"No addresses found for hostname: {hostname}")
    for addrinfo in addrinfos:
        ip_str = str(addrinfo[4][0])
        if _is_private_ip(ip_str):
            raise _FetchError("ssrf_blocked", f"Access to private/internal IP is not allowed: {hostname} -> {ip_str}")
    return str(addrinfos[0][4][0])


class _PinnedDNSNetworkBackend(httpcore.AsyncNetworkBackend):
    """httpcore backend that pins DNS resolution to a pre-validated IP."""

    def __init__(self, pinned_hosts: Dict[str, str]):
        self._pinned_hosts = pinned_hosts
        self._default_backend = httpcore.AnyIOBackend()

    async def connect_tcp(self, host: str, port: int, timeout: Optional[float] = None,
                          local_address: Optional[str] = None, socket_options=None) -> httpcore.AsyncNetworkStream:
        actual_host = self._pinned_hosts.get(host, host)
        return await self._default_backend.connect_tcp(actual_host, port, timeout=timeout,
                                                       local_address=local_address, socket_options=socket_options)

    async def connect_unix_socket(self, path: str, timeout: Optional[float] = None,
                                  socket_options=None) -> httpcore.AsyncNetworkStream:
        return await self._default_backend.connect_unix_socket(path, timeout=timeout, socket_options=socket_options)

    async def sleep(self, seconds: float) -> None:
        await self._default_backend.sleep(seconds)


def _create_pinned_transport(hostname: str, pinned_ip: str) -> httpx.AsyncHTTPTransport:
    """Create httpx transport with DNS pinned to validated IP."""
    backend = _PinnedDNSNetworkBackend({hostname: pinned_ip})
    transport = httpx.AsyncHTTPTransport()
    transport._pool._network_backend = backend
    return transport


async def _pre_request_ssrf_check(request: httpx.Request) -> None:
    """Event hook: re-validate SSRF before each outbound request (catches redirects)."""
    hostname = request.url.host
    if hostname:
        host_str = hostname.decode() if isinstance(hostname, bytes) else hostname
        try:
            addrinfos = socket.getaddrinfo(host_str, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        except socket.gaierror:
            raise _FetchError("url_not_accessible", f"Cannot resolve hostname: {host_str}")
        for addrinfo in addrinfos:
            ip_str = str(addrinfo[4][0])
            if _is_private_ip(ip_str):
                raise _FetchError("ssrf_blocked", f"DNS rebinding detected: {host_str} -> {ip_str}")


class _FetchError(Exception):
    def __init__(self, error_code: str, message: str = ""):
        self.error_code = error_code
        self.message = message
        super().__init__(f"{error_code}: {message}")


# --- HTML helpers ---

def _html_to_text(html: str) -> str:
    """Convert HTML to plain text (regex-based, no external deps)."""
    text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    text = re.sub(r'<(?:br|hr)[^>]*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<(?:/p|/div|/h[1-6]|/li|/tr|/section|/article)>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<(?:p|div|h[1-6]|li|tr|section|article)[^>]*>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    text = html_module.unescape(text)
    text = re.sub(r'[^\S\n]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _extract_title(html: str) -> str:
    match = re.search(r'<title[^>]*>(.*?)</title>', html, re.DOTALL | re.IGNORECASE)
    return html_module.unescape(match.group(1).strip()) if match else ""


# --- Fetch execution ---

_FETCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; BedrockProxy/1.0)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
_TEXT_TYPES = {
    "text/html", "text/plain", "text/xml", "application/xml",
    "application/xhtml+xml", "application/json", "text/csv", "text/markdown",
}


def _validate_url(url: str) -> None:
    if not url or not url.startswith(("http://", "https://")):
        raise _FetchError("invalid_input", f"Invalid URL: {url}")
    if len(url) > 250:
        raise _FetchError("url_too_long", "URL exceeds 250 characters")
    _validate_url_ssrf(url)


async def _execute_fetch(url: str, max_content_tokens: Optional[int] = None) -> Dict[str, Any]:
    """Fetch URL with SSRF-protected, DNS-pinned httpx client."""
    _validate_url(url)
    logger.info(f"[WebFetch] Fetching: {url}")

    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        raise _FetchError("invalid_input", f"Cannot extract hostname: {url}")

    pinned_ip = _resolve_and_validate(hostname)
    transport = _create_pinned_transport(hostname, pinned_ip)

    async def _validate_redirect(response: httpx.Response) -> None:
        if response.next_request is not None:
            redirect_url = str(response.next_request.url)
            try:
                _validate_url_ssrf(redirect_url)
            except _FetchError:
                raise _FetchError("ssrf_blocked", f"Redirect to blocked URL: {redirect_url}")

    client = httpx.AsyncClient(
        transport=transport, timeout=30.0, follow_redirects=True,
        event_hooks={"request": [_pre_request_ssrf_check], "response": [_validate_redirect]},
        headers=_FETCH_HEADERS,
    )
    try:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
        except _FetchError:
            raise
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "rate limit" in error_str:
                raise _FetchError("too_many_requests", str(e))
            raise _FetchError("url_not_accessible", str(e))

        content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
        is_pdf = content_type == "application/pdf" or url.lower().endswith(".pdf")

        if is_pdf:
            import base64
            content = base64.b64encode(resp.content).decode("utf-8")
            title = url.rsplit("/", 1)[-1] if "/" in url else url
            media_type = "application/pdf"
        elif content_type in _TEXT_TYPES or content_type.startswith("text/"):
            raw_html = resp.text
            title = _extract_title(raw_html) if "html" in content_type else ""
            content = _html_to_text(raw_html) if ("html" in content_type or "xml" in content_type) else raw_html
            media_type = "text/plain"
        else:
            raise _FetchError("unsupported_content_type", f"Content type not supported: {content_type}")

        # Token limit: 1 token ≈ 4 chars
        if max_content_tokens and content and not is_pdf:
            max_chars = max_content_tokens * 4
            if len(content) > max_chars:
                content = content[:max_chars]

        return {"url": str(resp.url), "title": title, "content": content, "media_type": media_type, "is_pdf": is_pdf}
    finally:
        await client.aclose()


# --- Tool definitions ---

_WEB_FETCH_TOOL = {
    "name": "web_fetch",
    "description": (
        "Fetch the full content of a web page or PDF document at a given URL. "
        "Returns the complete text content. Use this when you need to read the "
        "full content of a specific URL."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"url": {"type": "string", "description": "The URL to fetch content from"}},
        "required": ["url"],
    },
}


def _build_tools(original_tools: Optional[List[Any]]) -> List[Any]:
    """Replace web_fetch marker with custom tool definition."""
    if not original_tools:
        return [_WEB_FETCH_TOOL]
    result = []
    for tool in original_tools:
        t = tool if isinstance(tool, dict) else (tool.model_dump() if hasattr(tool, "model_dump") else {})
        if t.get("type", "") in WEB_FETCH_TOOL_TYPES:
            continue
        result.append(t)
    result.append(_WEB_FETCH_TOOL)
    return result


def _extract_config(request: MessageRequest) -> Dict[str, Any]:
    """Extract web fetch config (max_uses, allowed_domains, blocked_domains, max_content_tokens)."""
    config: Dict[str, Any] = {
        "max_uses": settings.web_fetch_default_max_uses,
        "max_content_tokens": settings.web_fetch_default_max_content_tokens,
    }
    if not request.tools:
        return config
    for tool in request.tools:
        t = tool if isinstance(tool, dict) else (tool.model_dump() if hasattr(tool, "model_dump") else {})
        if t.get("type", "") in WEB_FETCH_TOOL_TYPES:
            config["max_uses"] = t.get("max_uses") or settings.web_fetch_default_max_uses
            config["allowed_domains"] = t.get("allowed_domains")
            config["blocked_domains"] = t.get("blocked_domains")
            config["max_content_tokens"] = t.get("max_content_tokens") or settings.web_fetch_default_max_content_tokens
            break
    return config


def _filter_beta(anthropic_beta: Optional[str]) -> Optional[str]:
    if not anthropic_beta:
        return None
    filtered = [h.strip() for h in anthropic_beta.split(",") if h.strip() not in WEB_FETCH_BETA_HEADERS]
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


def _to_server_id(original_id: str) -> str:
    if original_id.startswith("srvtoolu_"):
        return original_id
    if original_id.startswith("toolu_"):
        return "srvtoolu_" + original_id[6:]
    return f"srvtoolu_{original_id}"


def _convert_to_server_tool_use(content: list) -> list:
    """Convert web_fetch tool_use blocks to server_tool_use."""
    converted = []
    for block in content:
        bd = block if isinstance(block, dict) else (block.model_dump() if hasattr(block, "model_dump") else {})
        if bd.get("type") == "tool_use" and bd.get("name") == "web_fetch":
            converted.append({
                "type": "server_tool_use",
                "id": _to_server_id(bd.get("id", "")),
                "name": "web_fetch",
                "input": bd.get("input", {}),
            })
        else:
            converted.append(bd)
    return converted


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_web_fetch_result(tool_use_id: str, fetch_data: Dict[str, Any]) -> Dict[str, Any]:
    source_type = "base64" if fetch_data.get("is_pdf") else "text"
    media_type = fetch_data.get("media_type", "text/plain")
    document: Dict[str, Any] = {
        "type": "document",
        "source": {"type": source_type, "media_type": media_type, "data": fetch_data.get("content", "")},
    }
    title = fetch_data.get("title", "")
    if title:
        document["title"] = title
    return {
        "type": "web_fetch_tool_result",
        "tool_use_id": tool_use_id,
        "content": {"type": "web_fetch_result", "url": fetch_data.get("url", ""), "content": document, "retrieved_at": _now_iso()},
    }


def _build_web_fetch_error(tool_use_id: str, error_code: str) -> Dict[str, Any]:
    return {
        "type": "web_fetch_tool_result",
        "tool_use_id": tool_use_id,
        "content": {"type": "web_fetch_tool_error", "error_code": error_code},
    }


def _check_domain_allowed(url: str, config: Dict[str, Any]) -> bool:
    """Check if URL domain is allowed by config."""
    allowed = config.get("allowed_domains")
    blocked = config.get("blocked_domains")
    if not allowed and not blocked:
        return True
    parsed = urlparse(url)
    domain = parsed.hostname
    if not domain:
        return False
    domain = domain.lower()
    if blocked:
        for bd in blocked:
            if domain == bd.lower() or domain.endswith("." + bd.lower()):
                return False
    if allowed:
        for ad in allowed:
            if domain == ad.lower() or domain.endswith("." + ad.lower()):
                return True
        return False  # not in allowed list
    return True


def _build_continuation_messages(
    messages: list,
    response_content: list,
    tool_results: list,
    document_registry: Dict[int, Dict[str, str]],
) -> list:
    """Build messages for next iteration with numbered documents."""
    new_messages = list(messages)

    assistant_content = []
    for block in response_content:
        bd = block if isinstance(block, dict) else (block.model_dump() if hasattr(block, "model_dump") else {})
        assistant_content.append(bd)
    new_messages.append({"role": "assistant", "content": assistant_content})

    user_content = []
    for result in tool_results:
        tool_use_id = result.get("tool_use_id", "")
        result_content = result.get("content", {})

        if result.get("type") == "web_fetch_tool_result" and isinstance(result_content, dict):
            content_type = result_content.get("type", "")
            if content_type == "web_fetch_result":
                url = result_content.get("url", "")
                doc = result_content.get("content", {})
                title = doc.get("title", "") if isinstance(doc, dict) else ""
                source = doc.get("source", {}) if isinstance(doc, dict) else {}
                content_data = source.get("data", "") if isinstance(source, dict) else ""

                idx = len(document_registry) + 1
                document_registry[idx] = {"url": url, "title": title, "content": content_data}
                result_text = (
                    f"[Document {idx}]\nTitle: {title}\nURL: {url}\nContent:\n{content_data}"
                    + _CITATION_REMINDER
                )
                user_content.append({"type": "tool_result", "tool_use_id": tool_use_id, "content": result_text})
            elif content_type == "web_fetch_tool_error":
                user_content.append({
                    "type": "tool_result", "tool_use_id": tool_use_id,
                    "content": f"Error: {result_content.get('error_code', 'unknown')}", "is_error": True,
                })
            else:
                user_content.append({"type": "tool_result", "tool_use_id": tool_use_id, "content": str(result_content)})
        else:
            user_content.append({"type": "tool_result", "tool_use_id": tool_use_id, "content": str(result_content)})

    new_messages.append({"role": "user", "content": user_content})
    return new_messages


def _post_process_citations(content_blocks: list, document_registry: Dict[int, Dict[str, str]]) -> list:
    """Convert [N] markers in text blocks to char_location citation objects."""
    if not document_registry:
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
                info = document_registry.get(idx)
                if not info:
                    continue
                source_content = info.get("content", "")
                citations.append({
                    "type": "char_location",
                    "document_index": idx - 1,
                    "document_title": info.get("title", ""),
                    "start_char_index": 0,
                    "end_char_index": min(len(source_content), 150),
                    "cited_text": source_content[:150],
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

async def handle_web_fetch(
    request: MessageRequest,
    request_id: str,
    anthropic_beta: Optional[str] = None,
    backend: Optional[LLMBackend] = None,
) -> MessageResponse:
    """Non-streaming web fetch agentic loop.

    ``backend`` drives every iteration's non-streaming model call. Falls back
    to the default Bedrock backend when ``None`` so legacy callers keep
    working.
    """
    if backend is None:
        raise ValueError("backend must be provided")

    config = _extract_config(request)
    max_uses = config["max_uses"]
    max_content_tokens = config.get("max_content_tokens")
    filtered_beta = _filter_beta(anthropic_beta)
    document_registry: Dict[int, Dict[str, str]] = {}
    all_content: List[Any] = []
    total_input = total_output = fetch_count = 0
    total_cache_creation = total_cache_read = 0
    messages: list = [
        m.model_dump(exclude_none=True) if hasattr(m, "model_dump") else m
        for m in request.messages
    ]

    for iteration in range(MAX_ITERATIONS):
        logger.info(f"[WebFetch] Iteration {iteration+1}, fetches={fetch_count}/{max_uses}")

        iter_request = MessageRequest(
            model=request.model,
            messages=messages,
            max_tokens=request.max_tokens,
            system=_inject_citation_system(request.system),
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

        response = await backend.invoke(iter_request, f"{request_id}_iter{iteration}", filtered_beta)
        total_input += response.usage.input_tokens
        total_output += response.usage.output_tokens
        total_cache_creation += response.usage.cache_creation_input_tokens or 0
        total_cache_read += response.usage.cache_read_input_tokens or 0
        response_content = response.content or []

        web_fetch_uses = [
            b for b in response_content
            if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") == "web_fetch"
        ]

        converted = _convert_to_server_tool_use(response_content)

        if not web_fetch_uses or response.stop_reason != "tool_use":
            # Final iteration: keep full content
            all_content.extend(converted)
            break

        # If quota already exhausted, add only text blocks and stop cleanly
        if fetch_count >= max_uses:
            logger.info(f"[WebFetch] max_uses ({max_uses}) exhausted, returning accumulated results")
            for b in converted:
                if isinstance(b, dict) and b.get("type") == "text":
                    all_content.append(b)
            break

        # Intermediate iteration: only accumulate server_tool_use blocks.
        # Text blocks from mid-loop turns are intentionally dropped — otherwise
        # Claude's next turn (which sees them in its history) tends to restate
        # them, producing duplicated near-identical paragraphs in the final output.
        for b in converted:
            if isinstance(b, dict) and b.get("type") == "server_tool_use":
                all_content.append(b)

        continuation_results = []
        for tool_use in web_fetch_uses:
            original_id = tool_use.get("id", "")
            server_id = _to_server_id(original_id)
            url = tool_use.get("input", {}).get("url", "")

            if fetch_count >= max_uses:
                client_result = _build_web_fetch_error(server_id, "max_uses_exceeded")
                cont_result = _build_web_fetch_error(original_id, "max_uses_exceeded")
            elif not _check_domain_allowed(url, config):
                client_result = _build_web_fetch_error(server_id, "url_not_allowed")
                cont_result = _build_web_fetch_error(original_id, "url_not_allowed")
            else:
                try:
                    fetch_data = await _execute_fetch(url, max_content_tokens)
                    client_result = _build_web_fetch_result(server_id, fetch_data)
                    cont_result = _build_web_fetch_result(original_id, fetch_data)
                    fetch_count += 1
                except _FetchError as e:
                    logger.error(f"[WebFetch] Fetch failed: {e}")
                    client_result = _build_web_fetch_error(server_id, e.error_code)
                    cont_result = _build_web_fetch_error(original_id, e.error_code)
                except Exception as e:
                    logger.error(f"[WebFetch] Fetch failed (unexpected): {e}")
                    client_result = _build_web_fetch_error(server_id, "unavailable")
                    cont_result = _build_web_fetch_error(original_id, "unavailable")

            continuation_results.append(cont_result)
            all_content.append(client_result)

        messages = _build_continuation_messages(messages, response_content, continuation_results, document_registry)

    # Post-process citations
    all_content = _post_process_citations(all_content, document_registry)

    usage = Usage(
        input_tokens=total_input,
        output_tokens=total_output,
        cache_creation_input_tokens=total_cache_creation or None,
        cache_read_input_tokens=total_cache_read or None,
        server_tool_use={"web_fetch_requests": fetch_count} if fetch_count > 0 else None,
    )

    return MessageResponse(
        id=request_id,
        content=all_content,
        model=request.model,
        stop_reason=response.stop_reason if 'response' in dir() else "end_turn",
        usage=usage,
    )


async def stream_web_fetch(
    request: MessageRequest,
    request_id: str,
    anthropic_beta: Optional[str] = None,
    backend: Optional[LLMBackend] = None,
) -> AsyncGenerator[str, None]:
    """Streaming web fetch — hybrid approach (non-streaming backend, SSE per iteration)."""
    # Emit an immediate "connect" ping to prime middlebox buffers and signal liveness.
    yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"
    # Run the underlying (non-streaming) work as a task and emit keep-alive pings
    # while we wait, so clients / CloudFront don't time out on long runs.
    task = asyncio.create_task(handle_web_fetch(request, request_id, anthropic_beta, backend))
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=settings.stream_ping_interval_sec)
            if done:
                break
            yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"
    except (asyncio.CancelledError, GeneratorExit):
        # Client disconnected — cancel the underlying work to avoid orphan
        # iterations (up to MAX_ITERATIONS of Bedrock calls) holding memory.
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

    for idx, block in enumerate(response.content):
        bd = block if isinstance(block, dict) else (block.model_dump(exclude_none=True) if hasattr(block, "model_dump") else block)
        block_type = bd.get("type", "text")

        if block_type == "text":
            # Per Anthropic SSE protocol, content_block_start for text carries
            # only an empty shell; citations are streamed via content_block_delta.
            start_block: Dict[str, Any] = {"type": "text", "text": ""}
            if bd.get("citations"):
                start_block["citations"] = []
        elif block_type in ("server_tool_use", "web_fetch_tool_result"):
            start_block = bd
        else:
            start_block = bd

        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': idx, 'content_block': start_block})}\n\n"

        if block_type == "text":
            delta = {"type": "text_delta", "text": bd.get("text", "")}
            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': idx, 'delta': delta})}\n\n"
            for citation in bd.get("citations") or []:
                citation_delta = {"type": "citations_delta", "citation": citation}
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': idx, 'delta': citation_delta})}\n\n"

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
