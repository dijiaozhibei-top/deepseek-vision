"""Application configuration via environment variables."""
from typing import List, Optional
from typing_extensions import Annotated
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode


class Settings(BaseSettings):
    # Server
    port: int = Field(default=8000, alias="PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    debug_upstream: bool = Field(default=False, alias="DEBUG_UPSTREAM")

    # Auth: comma-separated list of allowed API keys.
    # Empty by default so the server can start without a .env file
    # (the dashboard at / is accessible; all API routes return 401).
    master_api_keys: Annotated[List[str], NoDecode] = Field(default="", alias="MASTER_API_KEY")

    @field_validator("master_api_keys", mode="before")
    @classmethod
    def _split_keys(cls, v: object) -> List[str]:
        if isinstance(v, str):
            return [k.strip() for k in v.split(",") if k.strip()]
        if isinstance(v, list):
            return [str(k).strip() for k in v if str(k).strip()]
        return []

    # Admin dashboard password (protects /status, /admin/*)
    admin_password: str = Field(default="123456", alias="ADMIN_PASSWORD")

    # DeepSeek upstream (Anthropic Messages-compatible endpoint)
    deepseek_api_key: Optional[str] = Field(default=None, alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(default="https://api.deepseek.com/anthropic", alias="DEEPSEEK_BASE_URL")
    # Comma-separated list of model IDs to expose (client-id == upstream-id).
    # Supports "client-id:upstream-id" syntax for aliasing.
    deepseek_models: str = Field(default="deepseek-v4-pro,deepseek-v4-flash", alias="DEEPSEEK_MODELS")

    # Optional extra Anthropic-compatible upstream
    extra_backend_name: Optional[str] = Field(default=None, alias="EXTRA_BACKEND_NAME")
    extra_backend_base_url: Optional[str] = Field(default=None, alias="EXTRA_BACKEND_BASE_URL")
    extra_backend_api_key: Optional[str] = Field(default=None, alias="EXTRA_BACKEND_API_KEY")
    extra_backend_models: Optional[str] = Field(default=None, alias="EXTRA_BACKEND_MODELS")

    # Upstream wire format: "anthropic" (Messages API) or "openai" (Chat Completions).
    # For openai, BASE_URL must be an OpenAI-style root (e.g. https://api.deepseek.com),
    # not the .../anthropic suffix used by the Messages endpoint.
    deepseek_format: str = Field(default="anthropic", alias="DEEPSEEK_FORMAT")
    extra_backend_format: str = Field(default="anthropic", alias="EXTRA_BACKEND_FORMAT")

    @field_validator("deepseek_format", "extra_backend_format", mode="before")
    @classmethod
    def _normalize_format(cls, v: object) -> str:
        value = str(v).strip().lower()
        if value not in ("anthropic", "openai"):
            raise ValueError(f"unsupported upstream format {v!r} (expected 'anthropic' or 'openai')")
        return value

    # Vision middleware: any OpenAI-compatible vision endpoint
    vision_base_url: Optional[str] = Field(default="https://dashscope.aliyuncs.com/compatible-mode/v1", alias="VISION_BASE_URL")
    vision_api_key: Optional[str] = Field(default=None, alias="VISION_API_KEY")
    vision_model: Optional[str] = Field(default="qwen3.6-flash", alias="VISION_MODEL")
    vision_prompt: str = Field(
        default="Describe this image in detail. Be specific about text, objects, layout, and colors.",
        alias="VISION_PROMPT",
    )
    # Max number of images to process per request; additional images are passed through unchanged.
    vision_max_images: int = Field(default=5, alias="VISION_MAX_IMAGES")
    # Max output tokens for the vision model call. Thinking models (e.g. Mimo)
    # spend output budget on reasoning before the description, so the cap must
    # leave headroom or the returned content can be empty.
    vision_max_tokens: int = Field(default=2048, alias="VISION_MAX_TOKENS")
    # Per-image description request timeout. Large images / slow vision endpoints
    # can take a while; a too-short timeout silently degrades to "[Image N: description unavailable]".
    vision_timeout: int = Field(default=120, alias="VISION_TIMEOUT")

    # Request timeouts (seconds)
    upstream_timeout: int = Field(default=900, alias="UPSTREAM_TIMEOUT")
    upstream_stream_timeout: int = Field(default=1200, alias="UPSTREAM_STREAM_TIMEOUT")

    # Web Search
    web_search_provider: str = Field(default="tavily", alias="WEB_SEARCH_PROVIDER")
    tavily_api_key: Optional[str] = Field(default=None, alias="TAVILY_API_KEY")
    brave_api_key: Optional[str] = Field(default=None, alias="BRAVE_API_KEY")
    web_search_max_results: int = Field(default=5, alias="WEB_SEARCH_MAX_RESULTS")
    web_search_default_max_uses: int = Field(default=3, alias="WEB_SEARCH_DEFAULT_MAX_USES")

    # Web Fetch
    web_fetch_default_max_uses: int = Field(default=5, alias="WEB_FETCH_DEFAULT_MAX_USES")
    web_fetch_default_max_content_tokens: int = Field(default=100000, alias="WEB_FETCH_DEFAULT_MAX_CONTENT_TOKENS")

    # Streaming heartbeat
    stream_ping_interval_sec: int = Field(default=10, alias="STREAM_PING_INTERVAL_SEC")
    stream_gap_warn_sec: int = Field(default=10, alias="STREAM_GAP_WARN_SEC")

    # Diagnostics: tracemalloc frame depth. 0 disables.
    tracemalloc_frames: int = Field(default=0, alias="TRACEMALLOC_FRAMES")

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
