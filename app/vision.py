"""Vision middleware.

Intercepts image content blocks in incoming requests and converts them to text
descriptions via a configurable OpenAI-compatible vision endpoint. This lets
any text-only model (e.g. DeepSeek-Chat, DeepSeek-Reasoner) handle image inputs.

When VISION_* env vars are not set, this module is a no-op: image blocks are
passed through unchanged (useful when the upstream already supports vision).

Architecture (mirrors web_search two-pass pattern):
  1. Scan all message content blocks for images.
  2. Call vision model in parallel via asyncio.gather for each image.
  3. Replace each image block with a text block: "[Image N] <caption>".
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

import httpx

from app.config import settings
from app.schemas import MessageRequest

logger = logging.getLogger(__name__)


def _vision_enabled() -> bool:
    return bool(settings.vision_base_url and settings.vision_api_key and settings.vision_model)


async def _describe_image(
    image_data: str,
    media_type: Optional[str],
    image_url: Optional[str],
    index: int,
) -> str:
    """Call the vision provider and return a text description."""
    base_url = settings.vision_base_url.rstrip("/")
    headers = {
        "Authorization": f"Bearer {settings.vision_api_key}",
        "Content-Type": "application/json",
    }

    if image_url:
        image_content: Dict[str, Any] = {"type": "image_url", "image_url": {"url": image_url}}
    else:
        # base64 data URI
        mtype = media_type or "image/jpeg"
        image_content = {
            "type": "image_url",
            "image_url": {"url": f"data:{mtype};base64,{image_data}"},
        }

    body = {
        "model": settings.vision_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": settings.vision_prompt},
                    image_content,
                ],
            }
        ],
        "max_tokens": settings.vision_max_tokens,
    }

    # The vision endpoint can intermittently 500 (transient load). One retry
    # measurably improves the chance of getting a usable description.
    last_error: Optional[Exception] = None
    for attempt in range(2):
        try:
            async with httpx.AsyncClient(timeout=settings.vision_timeout) as client:
                resp = await client.post(f"{base_url}/chat/completions", headers=headers, json=body)
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    # Thinking vision models can exhaust the output budget on reasoning
                    # and leave content empty; fall back to their reasoning field.
                    msg = data["choices"][0].get("message") or {}
                    content = msg.get("reasoning") or msg.get("reasoning_content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
                last_error = "empty description"
                logger.warning(f"[Vision] image {index} attempt {attempt + 1} returned empty description")
        except Exception as e:
            last_error = e
            logger.warning(f"[Vision] image {index} attempt {attempt + 1} failed: {e}")
        if attempt == 0:
            await asyncio.sleep(1.0)
    if last_error:
        logger.error(f"[Vision] image {index} description failed after retry: {last_error}")
    return "description unavailable"


_IMAGE_SYSTEM_PROMPT = (
    "用户已在上方对话中上传图片，图片内容由视觉模型解析后以内联的 [Image N] "
    "文本块给出。请把每个 [Image N] 块当作你亲自看到的图像内容，直接依据它作答；"
    "不要声明你看不到图片，也不要声称你在凭一段文字描述猜测。"
)

_UNAVAILABLE_SYSTEM_PROMPT = (
    "用户上传的图片未能成功解析，当前对话没有可用的图片内容。"
    "请如实告知用户图片无法处理，并建议其重新上传；不要凭空猜测图片内容。"
)


def _with_image_guidance(system, guide: str) -> List[Dict[str, Any]]:
    """Append an image-handling instruction to the system prompt."""
    block = {"type": "text", "text": guide}
    if system is None:
        return [block]
    if isinstance(system, str):
        return [{"type": "text", "text": system}, block]
    if isinstance(system, list):
        out: List[Dict[str, Any]] = []
        for item in system:
            if hasattr(item, "model_dump"):
                out.append(item.model_dump(exclude_none=True))
            elif isinstance(item, dict):
                out.append(item)
            else:
                out.append({"type": "text", "text": str(item)})
        out.append(block)
        return out
    return [{"type": "text", "text": str(system)}, block]


def _extract_images(messages: List[Any]) -> List[Dict[str, Any]]:
    """Find all image blocks across all messages, recording their location.

    Covers images nested inside tool_result content — a common way tools hand
    images back to the model (screenshots as base64/image_url parts). `kind` is
    "block" for a top-level image block or "tool_result" for one nested inside a
    tool_result's content list.
    """
    found = []
    for msg_idx, msg in enumerate(messages):
        content = msg.get("content", []) if isinstance(msg, dict) else []
        if isinstance(content, str):
            continue
        for block_idx, block in enumerate(content):
            b = block if isinstance(block, dict) else (block.model_dump() if hasattr(block, "model_dump") else {})
            if b.get("type") == "image":
                source = b.get("source", {})
                found.append({
                    "msg_idx": msg_idx,
                    "block_idx": block_idx,
                    "kind": "block",
                    "inner_idx": None,
                    "data": source.get("data"),
                    "media_type": source.get("media_type"),
                    "url": source.get("url"),
                    "image_index": len(found) + 1,
                })
            elif b.get("type") == "tool_result":
                inner = b.get("content")
                if isinstance(inner, list):
                    for inner_idx, iblk in enumerate(inner):
                        ib = iblk if isinstance(iblk, dict) else (iblk.model_dump() if hasattr(iblk, "model_dump") else {})
                        if ib.get("type") == "image":
                            source = ib.get("source", {})
                            found.append({
                                "msg_idx": msg_idx,
                                "block_idx": block_idx,
                                "kind": "tool_result",
                                "inner_idx": inner_idx,
                                "data": source.get("data"),
                                "media_type": source.get("media_type"),
                                "url": source.get("url"),
                                "image_index": len(found) + 1,
                            })
    return found


async def maybe_apply_vision(request: MessageRequest) -> MessageRequest:
    """Replace image blocks with text descriptions if vision middleware is enabled."""
    if not _vision_enabled():
        return request

    messages_raw = [
        m.model_dump(exclude_none=True) if hasattr(m, "model_dump") else m
        for m in request.messages
    ]

    images = _extract_images(messages_raw)
    if not images:
        return request

    # Respect VISION_MAX_IMAGES: only describe up to the limit; pass extras through.
    max_imgs = settings.vision_max_images
    to_describe = images[:max_imgs]
    passthrough = images[max_imgs:]

    logger.info(
        f"[Vision] processing {len(to_describe)} image(s) via {settings.vision_model}"
        + (f", {len(passthrough)} passed through (limit={max_imgs})" if passthrough else "")
    )

    # Describe images in parallel
    tasks = [
        _describe_image(
            image_data=img["data"],
            media_type=img["media_type"],
            image_url=img["url"],
            index=img["image_index"],
        )
        for img in to_describe
    ]
    captions = await asyncio.gather(*tasks)

    # Substitute image blocks with text blocks in the raw message list,
    # whether the image sits at block level or nested inside a tool_result.
    for img, caption in zip(to_describe, captions):
        msg = messages_raw[img["msg_idx"]]
        content = msg.get("content", [])
        text_block = {
            "type": "text",
            "text": f"[Image {img['image_index']}] {caption}",
        }
        if img["kind"] == "tool_result":
            inner = content[img["block_idx"]].get("content")
            if isinstance(inner, list):
                inner[img["inner_idx"]] = text_block
        else:
            content[img["block_idx"]] = text_block

    # Steer the model: treat [Image N] blocks as the actual image and answer
    # directly, instead of announcing "I can't see the image".
    guidance = None
    if to_describe:
        ok = all(c != "description unavailable" for c in captions)
        guidance = _IMAGE_SYSTEM_PROMPT if ok else _UNAVAILABLE_SYSTEM_PROMPT

    return MessageRequest(
        model=request.model,
        messages=messages_raw,
        max_tokens=request.max_tokens,
        system=_with_image_guidance(request.system, guidance) if guidance else request.system,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stop_sequences=request.stop_sequences,
        stream=request.stream,
        tools=request.tools,
        tool_choice=request.tool_choice,
        thinking=request.thinking,
        metadata=request.metadata,
        output_config=request.output_config,
        context_management=request.context_management,
    )
