"""Pydantic models for Anthropic Messages API."""
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator


# --- Content Blocks ---

class CacheControl(BaseModel):
    type: Literal["ephemeral"] = "ephemeral"
    ttl: Optional[Literal["5m", "1h"]] = None


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str
    citations: Optional[List[Any]] = None
    cache_control: Optional[CacheControl] = None

    def model_dump(self, **kwargs):
        d = super().model_dump(**kwargs)
        if d.get("citations") is None:
            d.pop("citations", None)
        return d


class ImageSource(BaseModel):
    type: Literal["base64", "url"] = "base64"
    media_type: Optional[Literal["image/jpeg", "image/png", "image/gif", "image/webp"]] = None
    data: Optional[str] = None
    url: Optional[str] = None


class ImageContent(BaseModel):
    type: Literal["image"] = "image"
    source: ImageSource
    cache_control: Optional[CacheControl] = None


class DocumentSource(BaseModel):
    type: Literal["base64"] = "base64"
    media_type: Literal["application/pdf"]
    data: str


class DocumentContent(BaseModel):
    type: Literal["document"] = "document"
    source: DocumentSource
    cache_control: Optional[CacheControl] = None


class ThinkingContent(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: Optional[str] = None


class RedactedThinkingContent(BaseModel):
    type: Literal["redacted_thinking"] = "redacted_thinking"
    data: str


class CompactionContent(BaseModel):
    type: Literal["compaction"] = "compaction"
    content: Optional[str] = None


class ToolUseContent(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: Dict[str, Any]


class ToolResultContent(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Union[str, List[Any]]
    is_error: Optional[bool] = None
    cache_control: Optional[CacheControl] = None


ContentBlock = Union[
    TextContent, ImageContent, DocumentContent,
    ThinkingContent, RedactedThinkingContent, CompactionContent,
    ToolUseContent, ToolResultContent,
]


# --- Messages ---

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[ContentBlock]]

    @field_validator("content", mode="before")
    @classmethod
    def convert_string_to_list(cls, v):
        if isinstance(v, str):
            return [{"type": "text", "text": v}]
        return v


class SystemMessage(BaseModel):
    type: Literal["text"] = "text"
    text: str
    cache_control: Optional[CacheControl] = None


class Metadata(BaseModel):
    user_id: Optional[str] = None


# --- Request ---

class MessageRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: int = Field(default=4096, ge=1)
    system: Optional[Union[str, List[SystemMessage]]] = None
    temperature: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=1)
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    tools: Optional[List[Any]] = None
    tool_choice: Optional[Union[Literal["auto", "any"], Dict[str, str]]] = None
    thinking: Optional[Dict[str, Any]] = None
    metadata: Optional[Metadata] = None
    output_config: Optional[Dict[str, Any]] = None
    context_management: Optional[Dict[str, Any]] = None

    @field_validator("system", mode="before")
    @classmethod
    def convert_system_string_to_list(cls, v):
        if isinstance(v, str):
            return [{"type": "text", "text": v}]
        return v


# --- Response ---

class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None
    iterations: Optional[List[Dict[str, Any]]] = None
    server_tool_use: Optional[Dict[str, Any]] = None


class MessageResponse(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: List[Any]
    model: str
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None
    usage: Usage


# --- Count Tokens ---

class CountTokensRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemMessage]]] = None
    tools: Optional[List[Any]] = None

    @field_validator("system", mode="before")
    @classmethod
    def convert_system_string_to_list(cls, v):
        if isinstance(v, str):
            return [{"type": "text", "text": v}]
        return v


class CountTokensResponse(BaseModel):
    input_tokens: int
