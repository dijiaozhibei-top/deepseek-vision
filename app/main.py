"""FastAPI application."""
import logging
import os
import re
import tracemalloc
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.messages import router as messages_router
from app.models import router as models_router

# On Windows, Python's mimetypes reads the registry, which maps .js to
# text/plain. Browsers refuse to run <script type="module"> without a proper
# JavaScript MIME type, leaving the dashboard a blank white page.
import mimetypes
mimetypes.add_type("application/javascript", ".js")

_log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


class _AnsiStrippingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return _ANSI_ESCAPE.sub("", super().format(record))


_console = logging.StreamHandler()
_console.setFormatter(_formatter)

os.makedirs("logs", exist_ok=True)
from logging.handlers import TimedRotatingFileHandler
_file_handler = TimedRotatingFileHandler(
    "logs/app.log", when="midnight", interval=1, backupCount=1, encoding="utf-8"
)
_file_handler.setFormatter(
    _AnsiStrippingFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)

logging.basicConfig(level=_log_level, handlers=[_console, _file_handler])


class _Drop404Filter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        return " 404 " not in msg


logging.getLogger("uvicorn.access").addFilter(_Drop404Filter())

_scan_logger = logging.getLogger("app.scan")

# Routes accessible without admin token
ALLOWED_ROUTES: set[tuple[str, str]] = {
    ("GET", "/health"),
    ("POST", "/admin/login"),       # login itself is open
    ("GET", "/v1/models"),
    ("POST", "/v1/messages"),
    ("POST", "/v1/messages/count_tokens"),
    ("POST", "/v1/chat/completions"),
    ("POST", "/api/event_logging/batch"),
    ("GET", "/"),
}

# Routes that require a valid admin token
_ADMIN_ROUTES: set[tuple[str, str]] = {
    ("GET", "/status"),
    ("POST", "/admin/apply"),
}

_SCAN_PATTERN = re.compile(
    r"("
    r"\.env"
    r"|\.git"
    r"|\.(?:php|asp|aspx|jsp|sql|bak|old|swp)"
    r"|\.(?:tar\.gz|zip|rar|7z)"
    r"|/wp-(?:admin|login|content|includes)"
    r"|phpmyadmin"
    r"|/\.(?:ssh|aws|docker|npmrc|htpasswd)"
    r"|/actuator|/_next/|/api/heartbeat"
    r"|/boaform|/manager/html|/jenkins|/owa"
    r")",
    re.IGNORECASE,
)


def _classify(path: str) -> str:
    return "scan" if _SCAN_PATTERN.search(path) else "unknown"


def _resolve_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    real = request.headers.get("x-real-ip")
    if real:
        return real.strip()
    return request.client.host if request.client else "-"


@asynccontextmanager
async def lifespan(app: FastAPI):
    frames = settings.tracemalloc_frames
    if frames > 0 and not tracemalloc.is_tracing():
        tracemalloc.start(frames)
        print(f"  tracemalloc: enabled ({frames} frames)")
    else:
        print("  tracemalloc: disabled")

    print("Starting deepseek-vision proxy")
    print(f"  Port: {settings.port}")
    yield
    print("Shutting down")
    if tracemalloc.is_tracing():
        tracemalloc.stop()


app = FastAPI(
    title="deepseek-vision",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

_ALL_ALLOWED = ALLOWED_ROUTES | _ADMIN_ROUTES


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    method = request.method
    path = request.url.path

    if method == "OPTIONS":
        resp = await call_next(request)
        return _add_security_headers(resp)

    if method == "GET" and path.startswith("/assets/"):
        resp = await call_next(request)
        return _add_security_headers(resp)

    # Static files: icon and other public assets
    if method == "GET" and path in ("/icon.png", "/favicon.ico"):
        resp = await call_next(request)
        return _add_security_headers(resp)

    if (method, path) in ALLOWED_ROUTES:
        resp = await call_next(request)
        return _add_security_headers(resp)

    if (method, path) in _ADMIN_ROUTES:
        from app.admin_auth import verify_token
        auth = request.headers.get("authorization", "")
        if not (auth.startswith("Bearer ") and verify_token(auth[7:])):
            return JSONResponse(
                status_code=401,
                content={"type": "error", "error": {"type": "authentication_error", "message": "Admin login required"}},
                headers=_security_header_dict(),
            )
        resp = await call_next(request)
        return _add_security_headers(resp)

    if method == "HEAD":
        return Response(status_code=200, headers=_security_header_dict())

    ip = _resolve_ip(request)
    ua = request.headers.get("user-agent", "-")[:80]
    tag = _classify(path)
    _scan_logger.info(f'[SCAN] {method} {path} tag={tag} ip={ip} ua="{ua}"')
    return Response(status_code=404, content="Not Found", media_type="text/plain",
                    headers=_security_header_dict())


def _security_header_dict() -> dict:
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "no-referrer",
        "Content-Security-Policy": (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "   # inline styles for CSS-in-JS
            "img-src 'self' data:; "
            "connect-src 'self'; "
            "frame-ancestors 'none';"
        ),
    }


def _add_security_headers(response: Response) -> Response:
    for k, v in _security_header_dict().items():
        response.headers[k] = v
    return response


# CORS: restrict to same-origin for admin routes; open for API routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "x-api-key", "anthropic-beta"],
)

app.include_router(messages_router, prefix="/v1")
app.include_router(models_router, prefix="/v1")

from app.openai_compat import router as openai_router
app.include_router(openai_router, prefix="/v1")

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

if os.path.isdir(os.path.join(_STATIC_DIR, "assets")):
    app.mount("/assets", StaticFiles(directory=os.path.join(_STATIC_DIR, "assets")), name="assets")


@app.get("/icon.png")
async def icon():
    return FileResponse(os.path.join(_STATIC_DIR, "icon.png"), media_type="image/png")


@app.get("/")
async def ui():
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/admin/login")
async def admin_login(request: Request):
    from app.admin_auth import _check_rate_limit, issue_token
    ip = _resolve_ip(request)
    if not _check_rate_limit(ip):
        raise HTTPException(
            status_code=429,
            detail={"type": "rate_limit", "message": "Too many login attempts, wait 5 minutes"},
        )
    body = await request.json()
    password = body.get("password", "")
    import hmac
    if not hmac.compare_digest(str(password), settings.admin_password):
        raise HTTPException(
            status_code=401,
            detail={"type": "authentication_error", "message": "Invalid password"},
        )
    return {"token": issue_token()}


@app.post("/admin/apply")
async def admin_apply(request: Request):
    """Write submitted .env to disk, then restart. Auth enforced by middleware."""
    import asyncio
    import signal
    body = await request.json()
    env_text: str = body.get("env", "")
    if not isinstance(env_text, str):
        raise HTTPException(status_code=400, detail={"type": "bad_request", "message": "env must be a string"})

    # Refuse to write anything that looks like a path traversal or shell injection
    if any(c in env_text for c in ["\x00", "\r"]):
        raise HTTPException(status_code=400, detail={"type": "bad_request", "message": "Invalid content"})

    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    # Write atomically: temp file → rename
    tmp_path = env_path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(env_text)
        os.replace(tmp_path, env_path)
    except OSError as e:
        raise HTTPException(status_code=500, detail={"type": "io_error", "message": str(e)})
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    async def _restart():
        await asyncio.sleep(0.4)
        os.kill(os.getpid(), signal.SIGTERM)

    asyncio.create_task(_restart())
    return {"status": "restarting"}


@app.get("/status")
async def status():
    """Return current proxy state. Auth enforced by middleware."""
    from app.router import MODEL_REGISTRY
    models = list(MODEL_REGISTRY.keys())
    backends: dict = {}
    for model_id, backend in MODEL_REGISTRY.items():
        backends.setdefault(backend.name, []).append(model_id)

    return {
        "status": "ok",
        "models": models,
        "backends": [{"name": name, "models": mlist} for name, mlist in backends.items()],
        "vision": {
            "enabled": bool(settings.vision_base_url and settings.vision_api_key and settings.vision_model),
            "model": settings.vision_model or None,
        },
        "web_search": {
            "enabled": bool(
                (settings.web_search_provider == "tavily" and settings.tavily_api_key)
                or (settings.web_search_provider == "brave" and settings.brave_api_key)
            ),
            "provider": settings.web_search_provider,
        },
        "web_fetch": {"enabled": True},
    }


@app.post("/api/event_logging/batch")
async def event_logging():
    return {"status": "ok"}


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    if isinstance(exc.detail, dict):
        return JSONResponse(
            status_code=exc.status_code,
            content={"type": "error", "error": exc.detail},
            headers=_security_header_dict(),
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={"type": "error", "error": {"type": "api_error", "message": str(exc.detail)}},
        headers=_security_header_dict(),
    )
