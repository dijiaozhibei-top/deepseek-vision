"""Admin dashboard authentication.

Token scheme: HMAC-SHA256(admin_password, timestamp) with 24-hour expiry.
No external dependencies — stdlib only.
"""
import hmac
import time
from typing import Dict, List

from fastapi import HTTPException, Request

from app.config import settings

# Simple in-memory rate limiter: max 5 login attempts per IP per 5 minutes.
_attempts: Dict[str, List[float]] = {}


def _check_rate_limit(ip: str) -> bool:
    now = time.time()
    window = [t for t in _attempts.get(ip, []) if now - t < 300]
    if len(window) >= 5:
        _attempts[ip] = window
        return False
    window.append(now)
    _attempts[ip] = window
    return True


def issue_token() -> str:
    ts = str(int(time.time()))
    mac = hmac.new(settings.admin_password.encode(), ts.encode(), "sha256").hexdigest()
    return f"{ts}.{mac}"


def verify_token(token: str) -> bool:
    try:
        ts_str, mac = token.split(".", 1)
        if time.time() - int(ts_str) > 86400:   # 24-hour expiry
            return False
        expected = hmac.new(settings.admin_password.encode(), ts_str.encode(), "sha256").hexdigest()
        return hmac.compare_digest(mac, expected)
    except Exception:
        return False


async def require_admin(request: Request) -> None:
    """FastAPI dependency — raises 401 if no valid admin token."""
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer ") and verify_token(auth[7:]):
        return
    raise HTTPException(
        status_code=401,
        detail={"type": "authentication_error", "message": "Admin token required"},
    )
