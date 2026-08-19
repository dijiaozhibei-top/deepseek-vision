"""Abstract LLMBackend interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import HTTPException

from app.schemas import CountTokensRequest, MessageRequest, MessageResponse


class LLMBackend(ABC):
    name: str = "base"
    model_map: Dict[str, str] = {}

    def resolve_model_id(self, client_model: str) -> str:
        return self.model_map.get(client_model, client_model)

    def supports(self, client_model: str) -> bool:
        return client_model in self.model_map

    @abstractmethod
    async def invoke(
        self,
        request: MessageRequest,
        request_id: str,
        anthropic_beta: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> MessageResponse: ...

    @abstractmethod
    def stream(
        self,
        request: MessageRequest,
        request_id: str,
        anthropic_beta: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[str, None]: ...

    async def count_tokens(self, request: CountTokensRequest) -> int:
        raise HTTPException(
            status_code=501,
            detail={
                "type": "not_supported",
                "message": f"count_tokens is not supported for backend '{self.name}'",
            },
        )
