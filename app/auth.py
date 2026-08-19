"""MASTER_API_KEY authentication (supports multiple keys)."""
import hmac

from fastapi import Header, HTTPException

from app.config import settings


async def require_auth(x_api_key: str = Header(..., alias="x-api-key")) -> str:
    """Validate x-api-key against MASTER_API_KEY list (timing-safe)."""
    for key in settings.master_api_keys:
        if hmac.compare_digest(x_api_key, key):
            return x_api_key
    raise HTTPException(
        status_code=401,
        detail={"type": "authentication_error", "message": "Invalid API key"},
    )


async def require_auth_flexible(
    x_api_key: str = Header(None, alias="x-api-key"),
    authorization: str = Header(None, alias="authorization"),
) -> str:
    """Accept either x-api-key or Authorization: Bearer <key> (for OpenAI SDK compat)."""
    candidates = []
    if x_api_key:
        candidates.append(x_api_key)
    if authorization and authorization.startswith("Bearer "):
        candidates.append(authorization[7:])

    for candidate in candidates:
        for key in settings.master_api_keys:
            if hmac.compare_digest(candidate, key):
                return candidate

    raise HTTPException(
        status_code=401,
        detail={"type": "authentication_error", "message": "Invalid API key"},
    )
