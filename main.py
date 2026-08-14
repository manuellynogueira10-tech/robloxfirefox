"""
Roblox Browser Backend - single-file FastAPI + Playwright + Tavily backend.

Security model (secure-by-default):
- Requires X-API-Key on /search and /render (BROWSER_API_KEY env var).
- SSRF protections: only public HTTP(S) hosts, DNS/IP checks on every browser request,
  private/loopback/link-local/reserved/multicast addresses blocked, ports restricted.
- Fresh non-persistent Playwright BrowserContext per render.
- Service workers blocked so request interception cannot be bypassed by them.
- Downloads disabled; non-HTTP(S) network schemes blocked.
- Page POST/PUT/PATCH/DELETE blocked by default (ALLOW_PAGE_POST=1 to relax).
- In-memory request rate limit, concurrency limit and short TTL caches.
- No CORS middleware is enabled because Roblox HttpService does not need browser CORS.

Required packages:
    pip install fastapi "uvicorn[standard]" httpx playwright
    playwright install firefox

Required environment variables:
    BROWSER_API_KEY=<a long random secret>
    TAVILY_API_KEY=tvly-...          # required only for /search and text fallback

Useful optional environment variables:
    BROWSER_ENGINE=firefox           # firefox | chromium | webkit
    PORT=8000
    ALLOW_HTTP=0
    ALLOW_PAGE_POST=0
    ALLOW_WEBSOCKETS=0
    ENABLE_DOCS=0
    ALLOWED_PORTS=80,443
    RATE_LIMIT_PER_MINUTE=60
    MAX_CONCURRENT_PAGES=2
    NAV_TIMEOUT_MS=18000
    RENDER_SETTLE_MS=700
    RENDER_CACHE_TTL=60
    SEARCH_CACHE_TTL=30
    MAX_NODES=1400
    MAX_DEPTH=36
    MAX_TEXT_CHARS=1200

Run locally:
    python main.py

Roblox should send:
    Headers = {
        ["Content-Type"] = "application/json",
        ["X-API-Key"] = "YOUR_BROWSER_API_KEY",
    }

This backend serializes the *rendered DOM + computed styles* rather than returning raw
<script>/<style> source, which prevents the strange source-code text that appeared in
older HTML-tokenizer versions.
"""

from __future__ import annotations

import asyncio
import hmac
import ipaddress
import json
import os
import socket
import time
from collections import OrderedDict, defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Deque, Dict, Literal, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from playwright.async_api import (
    Browser,
    BrowserContext,
    Error as PlaywrightError,
    Playwright,
    Request as PlaywrightRequest,
    Route,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)


APP_NAME = "Roblox Browser API"
APP_VERSION = "3.0.0"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


BROWSER_API_KEY = os.getenv("BROWSER_API_KEY", "").strip()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
BROWSER_ENGINE = os.getenv("BROWSER_ENGINE", "firefox").strip().lower()
ALLOW_HTTP = _env_bool("ALLOW_HTTP", False)
ALLOW_PAGE_POST = _env_bool("ALLOW_PAGE_POST", False)
ALLOW_WEBSOCKETS = _env_bool("ALLOW_WEBSOCKETS", False)
ENABLE_DOCS = _env_bool("ENABLE_DOCS", False)
RATE_LIMIT_PER_MINUTE = _env_int("RATE_LIMIT_PER_MINUTE", 60, 5, 1000)
MAX_CONCURRENT_PAGES = _env_int("MAX_CONCURRENT_PAGES", 2, 1, 12)
NAV_TIMEOUT_MS = _env_int("NAV_TIMEOUT_MS", 18_000, 3_000, 60_000)
ACTION_TIMEOUT_MS = _env_int("ACTION_TIMEOUT_MS", 8_000, 1_000, 30_000)
RENDER_SETTLE_MS = _env_int("RENDER_SETTLE_MS", 700, 0, 5_000)
RENDER_CACHE_TTL = _env_float("RENDER_CACHE_TTL", 60.0, 0.0, 900.0)
SEARCH_CACHE_TTL = _env_float("SEARCH_CACHE_TTL", 30.0, 0.0, 900.0)
DNS_CACHE_TTL = _env_float("DNS_CACHE_TTL", 60.0, 5.0, 600.0)
MAX_NODES = _env_int("MAX_NODES", 1400, 100, 5000)
MAX_DEPTH = _env_int("MAX_DEPTH", 36, 8, 80)
MAX_TEXT_CHARS = _env_int("MAX_TEXT_CHARS", 1200, 100, 8000)
MAX_RESPONSE_BYTES = _env_int("MAX_RESPONSE_BYTES", 2_500_000, 200_000, 8_000_000)
PORT = _env_int("PORT", 8000, 1, 65535)

_allowed_ports_raw = os.getenv("ALLOWED_PORTS", "80,443")
ALLOWED_PORTS: set[int] = set()
for part in _allowed_ports_raw.split(","):
    try:
        p = int(part.strip())
        if 1 <= p <= 65535:
            ALLOWED_PORTS.add(p)
    except ValueError:
        pass
if not ALLOWED_PORTS:
    ALLOWED_PORTS = {80, 443}

BLOCKED_HOST_SUFFIXES = (
    ".localhost",
    ".local",
    ".internal",
    ".home",
    ".lan",
)

# Media streaming and event streams are unnecessary for DOM reconstruction and can
# consume a lot of bandwidth/resources. XHR/fetch remain enabled so modern pages can hydrate.
BLOCKED_RESOURCE_TYPES = {"media", "eventsource"}
SAFE_PAGE_METHODS = {"GET", "HEAD", "OPTIONS"}


class Viewport(BaseModel):
    width: int = Field(default=1280, ge=320, le=2560)
    height: int = Field(default=720, ge=300, le=1800)


class RenderRequestModel(BaseModel):
    url: str = Field(min_length=1, max_length=2048)
    viewport: Viewport = Field(default_factory=Viewport)

    @field_validator("url")
    @classmethod
    def strip_url(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("URL vazia")
        return value


class SearchRequestModel(BaseModel):
    query: str = Field(min_length=1, max_length=300)
    max_results: int = Field(default=8, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("Pesquisa vazia")
        return value


class AsyncTTLCache:
    def __init__(self, max_items: int = 128) -> None:
        self.max_items = max_items
        self._items: "OrderedDict[str, Tuple[float, Any]]" = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any | None:
        now = time.monotonic()
        async with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            expires_at, value = item
            if expires_at <= now:
                self._items.pop(key, None)
                return None
            self._items.move_to_end(key)
            return value

    async def set(self, key: str, value: Any, ttl: float) -> None:
        if ttl <= 0:
            return
        async with self._lock:
            self._items[key] = (time.monotonic() + ttl, value)
            self._items.move_to_end(key)
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)


class SlidingWindowRateLimiter:
    def __init__(self, limit: int, window_seconds: float = 60.0) -> None:
        self.limit = limit
        self.window = window_seconds
        self._events: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def consume(self, key: str) -> Tuple[bool, int]:
        now = time.monotonic()
        cutoff = now - self.window
        async with self._lock:
            q = self._events[key]
            while q and q[0] <= cutoff:
                q.popleft()
            if len(q) >= self.limit:
                retry_after = max(1, int(self.window - (now - q[0])))
                return False, retry_after
            q.append(now)
            return True, 0


class PublicHostPolicy:
    """Resolve hosts and reject addresses that are not globally routable."""

    def __init__(self, ttl: float = 60.0) -> None:
        self.ttl = ttl
        self._cache: Dict[str, Tuple[float, bool]] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _literal_ip_is_public(host: str) -> Optional[bool]:
        try:
            ip = ipaddress.ip_address(host.strip("[]"))
        except ValueError:
            return None
        return ip.is_global

    async def is_public(self, host: str) -> bool:
        host = host.strip().rstrip(".").lower()
        if not host:
            return False

        if host == "localhost" or host.endswith(BLOCKED_HOST_SUFFIXES):
            return False

        literal = self._literal_ip_is_public(host)
        if literal is not None:
            return literal

        # Normalize international domain names before DNS lookup.
        try:
            host_ascii = host.encode("idna").decode("ascii")
        except UnicodeError:
            return False

        now = time.monotonic()
        async with self._lock:
            cached = self._cache.get(host_ascii)
            if cached and cached[0] > now:
                return cached[1]

        try:
            loop = asyncio.get_running_loop()
            infos = await asyncio.wait_for(
                loop.getaddrinfo(
                    host_ascii,
                    None,
                    family=socket.AF_UNSPEC,
                    type=socket.SOCK_STREAM,
                ),
                timeout=3.0,
            )
        except (asyncio.TimeoutError, socket.gaierror, OSError):
            result = False
        else:
            ips: set[str] = set()
            for info in infos:
                sockaddr = info[4]
                if sockaddr:
                    ips.add(str(sockaddr[0]))
            # Fail closed: every resolved address must be globally routable.
            result = bool(ips) and all(ipaddress.ip_address(ip).is_global for ip in ips)

        async with self._lock:
            self._cache[host_ascii] = (time.monotonic() + self.ttl, result)
        return result


@dataclass
class Runtime:
    playwright: Playwright
    browser: Browser
    http: httpx.AsyncClient
    host_policy: PublicHostPolicy
    limiter: SlidingWindowRateLimiter
    semaphore: asyncio.Semaphore
    render_cache: AsyncTTLCache
    search_cache: AsyncTTLCache
    browser_lock: asyncio.Lock


async def _launch_browser(pw: Playwright) -> Browser:
    if BROWSER_ENGINE == "firefox":
        browser_type = pw.firefox
    elif BROWSER_ENGINE == "chromium":
        browser_type = pw.chromium
    elif BROWSER_ENGINE == "webkit":
        browser_type = pw.webkit
    else:
        raise RuntimeError("BROWSER_ENGINE deve ser firefox, chromium ou webkit")

    # Do NOT add --no-sandbox here. Browser sandboxing is an important security layer.
    return await browser_type.launch(headless=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not BROWSER_API_KEY:
        raise RuntimeError(
            "BROWSER_API_KEY não configurada. Defina uma chave longa e aleatória antes de iniciar."
        )

    pw = await async_playwright().start()
    browser: Optional[Browser] = None
    http: Optional[httpx.AsyncClient] = None
    try:
        browser = await _launch_browser(pw)
        http = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=20.0, write=10.0, pool=5.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            follow_redirects=True,
            headers={"User-Agent": f"{APP_NAME}/{APP_VERSION}"},
        )
        app.state.runtime = Runtime(
            playwright=pw,
            browser=browser,
            http=http,
            host_policy=PublicHostPolicy(DNS_CACHE_TTL),
            limiter=SlidingWindowRateLimiter(RATE_LIMIT_PER_MINUTE),
            semaphore=asyncio.Semaphore(MAX_CONCURRENT_PAGES),
            render_cache=AsyncTTLCache(max_items=96),
            search_cache=AsyncTTLCache(max_items=128),
            browser_lock=asyncio.Lock(),
        )
        yield
    finally:
        if http is not None:
            await http.aclose()
        if browser is not None:
            try:
                await browser.close()
            except Exception:
                pass
        await pw.stop()


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if ENABLE_DOCS else None,
    redoc_url=None,
    openapi_url="/openapi.json" if ENABLE_DOCS else None,
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Frame-Options"] = "DENY"
    return response


def _runtime(request: Request) -> Runtime:
    return request.app.state.runtime


async def require_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    if not x_api_key or not hmac.compare_digest(x_api_key, BROWSER_API_KEY):
        raise HTTPException(status_code=401, detail="API key inválida")

    runtime = _runtime(request)
    # We intentionally use the immediate peer IP; we do not trust arbitrary X-Forwarded-For.
    peer = request.client.host if request.client else "unknown"
    allowed, retry_after = await runtime.limiter.consume(peer)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit da API. Tente novamente em {retry_after}s.",
            headers={"Retry-After": str(retry_after)},
        )


@app.get("/")
async def root(request: Request) -> Dict[str, Any]:
    runtime = _runtime(request)
    return {
        "success": True,
        "service": APP_NAME,
        "version": APP_VERSION,
        "browser_engine": BROWSER_ENGINE,
        "browser_connected": runtime.browser.is_connected(),
        "endpoints": ["POST /search", "POST /render", "GET /health"],
    }


@app.get("/health")
async def health(request: Request) -> Dict[str, Any]:
    runtime = _runtime(request)
    return {
        "ok": runtime.browser.is_connected(),
        "version": APP_VERSION,
        "browser": BROWSER_ENGINE,
    }


def _normalize_url(raw: str) -> str:
    raw = raw.strip()
    if "://" not in raw:
        raw = "https://" + raw

    parts = urlsplit(raw)
    scheme = parts.scheme.lower()
    if scheme not in {"https", "http"}:
        raise ValueError("Somente HTTP/HTTPS são permitidos")
    if scheme == "http" and not ALLOW_HTTP:
        raise ValueError("HTTP sem TLS está desativado; use HTTPS")
    if not parts.hostname:
        raise ValueError("URL sem host")
    if parts.username or parts.password:
        raise ValueError("URLs com usuário/senha não são permitidas")

    try:
        port = parts.port
    except ValueError as exc:
        raise ValueError("Porta inválida") from exc

    effective_port = port or (443 if scheme == "https" else 80)
    if effective_port not in ALLOWED_PORTS:
        raise ValueError(f"Porta {effective_port} não permitida")

    # Remove fragment; the page itself can still update the final URL later.
    return urlunsplit((scheme, parts.netloc, parts.path or "/", parts.query, ""))


async def _assert_public_url(runtime: Runtime, url: str) -> None:
    parts = urlsplit(url)
    host = parts.hostname or ""
    if not await runtime.host_policy.is_public(host):
        raise ValueError("Host privado/local/reservado ou DNS inválido")


async def _ensure_browser(runtime: Runtime) -> Browser:
    if runtime.browser.is_connected():
        return runtime.browser

    async with runtime.browser_lock:
        if runtime.browser.is_connected():
            return runtime.browser
        runtime.browser = await _launch_browser(runtime.playwright)
        return runtime.browser


async def _route_guard(runtime: Runtime, route: Route, request: PlaywrightRequest) -> None:
    try:
        resource_type = request.resource_type
        if resource_type in BLOCKED_RESOURCE_TYPES:
            await route.abort()
            return

        method = request.method.upper()
        if not ALLOW_PAGE_POST and method not in SAFE_PAGE_METHODS:
            await route.abort()
            return

        parts = urlsplit(request.url)
        scheme = parts.scheme.lower()

        # Local in-document schemes do not make an outbound network request.
        if scheme in {"data", "blob", "about"}:
            await route.continue_()
            return

        if scheme not in {"https", "http"}:
            await route.abort()
            return
        if scheme == "http" and not ALLOW_HTTP:
            await route.abort()
            return
        if not parts.hostname:
            await route.abort()
            return

        try:
            port = parts.port
        except ValueError:
            await route.abort()
            return
        effective_port = port or (443 if scheme == "https" else 80)
        if effective_port not in ALLOWED_PORTS:
            await route.abort()
            return

        if not await runtime.host_policy.is_public(parts.hostname):
            await route.abort()
            return

        await route.continue_()
    except Exception:
        # Fail closed: a routing exception must never result in an unrestricted request.
        try:
            await route.abort()
        except Exception:
            pass


SERIALIZE_DOM_JS = r"""
(opts) => {
  const MAX_NODES = opts.maxNodes;
  const MAX_DEPTH = opts.maxDepth;
  const MAX_TEXT = opts.maxText;

  const SKIP_TAGS = new Set([
    'SCRIPT','STYLE','NOSCRIPT','TEMPLATE','HEAD','TITLE','META','LINK','BASE',
    'IFRAME','OBJECT','EMBED','CANVAS','SVG','PATH','SOURCE','TRACK'
  ]);

  // Keep CSS property names in kebab-case so the existing Luau HTMLRenderer
  // can consume them directly.
  const STYLE_PROPS = [
    'display','visibility','position','top','right','bottom','left','z-index',
    'width','height','min-width','min-height','max-width','max-height',
    'margin-top','margin-right','margin-bottom','margin-left',
    'padding-top','padding-right','padding-bottom','padding-left',
    'background-color','background-image','color',
    'font-family','font-size','font-weight','font-style','line-height',
    'text-align','text-decoration-line','text-transform','white-space',
    'border-top-width','border-right-width','border-bottom-width','border-left-width',
    'border-top-color','border-right-color','border-bottom-color','border-left-color',
    'border-top-style','border-right-style','border-bottom-style','border-left-style',
    'border-top-left-radius','border-top-right-radius','border-bottom-right-radius','border-bottom-left-radius',
    'opacity','overflow-x','overflow-y',
    'flex-direction','flex-wrap','flex-grow','flex-shrink','flex-basis',
    'justify-content','align-items','align-content','align-self','gap','row-gap','column-gap',
    'grid-template-columns','grid-template-rows','object-fit'
  ];

  let count = 0;
  let truncated = false;

  function cleanText(text) {
    return String(text || '').replace(/\u00a0/g, ' ').replace(/\s+/g, ' ').trim().slice(0, MAX_TEXT);
  }

  function getAttrs(el) {
    const out = {};
    const allow = ['id','class','title','alt','role','aria-label','name','type','placeholder','method','action'];
    for (const name of allow) {
      const value = el.getAttribute && el.getAttribute(name);
      if (value) out[name] = String(value).slice(0, 1000);
    }

    if (el instanceof HTMLAnchorElement && el.href) out.href = el.href;
    if (el instanceof HTMLImageElement) {
      if (el.currentSrc || el.src) out.src = el.currentSrc || el.src;
      if (el.naturalWidth) out.naturalWidth = el.naturalWidth;
      if (el.naturalHeight) out.naturalHeight = el.naturalHeight;
    }
    if (el instanceof HTMLInputElement) {
      const t = (el.type || 'text').toLowerCase();
      out.type = t;
      if (t !== 'password' && t !== 'hidden' && el.value && el.value.length <= 500) {
        out.value = el.value;
      }
      if (el.checked) out.checked = true;
      if (el.disabled) out.disabled = true;
    }
    if (el instanceof HTMLTextAreaElement && el.value && el.value.length <= 1000) {
      out.value = el.value;
    }
    if (el instanceof HTMLButtonElement && el.disabled) out.disabled = true;
    return out;
  }

  function getStyle(el) {
    const cs = getComputedStyle(el);
    const out = {};
    for (const prop of STYLE_PROPS) {
      const value = cs.getPropertyValue(prop);
      if (value && value !== 'normal' && value !== 'none' && value !== 'auto' && value !== '0px') {
        out[prop] = value;
      }
    }
    out.display = cs.getPropertyValue('display');
    out.position = cs.getPropertyValue('position');
    out.visibility = cs.getPropertyValue('visibility');
    out.opacity = cs.getPropertyValue('opacity');
    const bg = cs.getPropertyValue('background-color');
    if (bg && bg !== 'rgba(0, 0, 0, 0)') out['background-color'] = bg;
    const color = cs.getPropertyValue('color');
    if (color) out.color = color;
    return out;
  }

  function elementVisible(el, style) {
    if (style.display === 'none' || style.visibility === 'hidden' || Number(style.opacity) === 0) return false;
    if (el instanceof HTMLInputElement && (el.type || '').toLowerCase() === 'hidden') return false;
    const rect = el.getBoundingClientRect();
    const hasSize = rect.width > 0 || rect.height > 0;
    const hasText = cleanText(el.textContent).length > 0;
    return hasSize || hasText;
  }

  function walk(node, depth) {
    if (count >= MAX_NODES) { truncated = true; return null; }
    if (depth > MAX_DEPTH) { truncated = true; return null; }

    if (node.nodeType === Node.TEXT_NODE) {
      const parent = node.parentElement;
      if (!parent || SKIP_TAGS.has(parent.tagName)) return null;
      const text = cleanText(node.nodeValue);
      if (!text) return null;
      count++;
      return { tag: '#text', text };
    }

    if (node.nodeType !== Node.ELEMENT_NODE) return null;
    const el = node;
    if (SKIP_TAGS.has(el.tagName)) return null;

    const style = getStyle(el);
    if (!elementVisible(el, style)) return null;

    count++;
    const rect = el.getBoundingClientRect();
    const out = {
      tag: el.tagName.toLowerCase(),
      attributes: getAttrs(el),
      style,
      rect: {
        x: Math.round(rect.x * 100) / 100,
        y: Math.round(rect.y * 100) / 100,
        width: Math.round(rect.width * 100) / 100,
        height: Math.round(rect.height * 100) / 100
      },
      children: []
    };

    for (const child of el.childNodes) {
      const serialized = walk(child, depth + 1);
      if (serialized) out.children.push(serialized);
      if (count >= MAX_NODES) { truncated = true; break; }
    }

    return out;
  }

  const body = document.body ? walk(document.body, 0) : null;
  let favicon = '';
  const icon = document.querySelector('link[rel~="icon"], link[rel="shortcut icon"]');
  if (icon && icon.href) favicon = icon.href;
  if (!favicon) {
    try { favicon = new URL('/favicon.ico', location.href).href; } catch (_) {}
  }

  return {
    title: document.title || 'Nova aba',
    url: location.href,
    favicon,
    viewport: { width: innerWidth, height: innerHeight },
    nodeCount: count,
    truncated,
    body
  };
}
"""


def _json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _truncate_tree_to_size(document: Dict[str, Any], max_bytes: int) -> Dict[str, Any]:
    """Last-resort response cap. Removes deepest tail nodes until payload fits."""
    if _json_size(document) <= max_bytes:
        return document

    document["truncated"] = True
    root = document.get("body")
    if not isinstance(root, dict):
        return document

    # Collect containers in breadth-first order; prune from the deepest/right-most.
    queue: list[Tuple[Dict[str, Any], int]] = [(root, 0)]
    containers: list[Tuple[Dict[str, Any], int]] = []
    while queue:
        node, depth = queue.pop(0)
        children = node.get("children")
        if isinstance(children, list) and children:
            containers.append((node, depth))
            for child in children:
                if isinstance(child, dict):
                    queue.append((child, depth + 1))

    containers.sort(key=lambda item: item[1], reverse=True)
    for node, _depth in containers:
        children = node.get("children")
        if not isinstance(children, list):
            continue
        while children and _json_size(document) > max_bytes:
            children.pop()
        if _json_size(document) <= max_bytes:
            break

    return document


async def _tavily_extract_fallback(runtime: Runtime, url: str, reason: str) -> Optional[Dict[str, Any]]:
    if not TAVILY_API_KEY:
        return None
    try:
        response = await runtime.http.post(
            "https://api.tavily.com/extract",
            headers={
                "Authorization": f"Bearer {TAVILY_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "urls": url,
                "extract_depth": "basic",
                "include_images": False,
                "include_favicon": True,
                "format": "text",
                "timeout": 10,
                "include_usage": False,
            },
        )
        if response.status_code != 200:
            return None
        data = response.json()
        results = data.get("results") or []
        if not results:
            return None
        item = results[0]
        raw = str(item.get("raw_content") or "").strip()
        if not raw:
            return None
        raw = raw[:30_000]
        paragraphs = [p.strip() for p in raw.splitlines() if p.strip()][:120]
        children: list[Dict[str, Any]] = [
            {
                "tag": "h1",
                "attributes": {},
                "style": {"font-size": "28px", "font-weight": "700"},
                "children": [{"tag": "#text", "text": "Modo de leitura"}],
            },
            {
                "tag": "p",
                "attributes": {},
                "style": {"color": "rgb(100, 100, 110)", "font-size": "13px"},
                "children": [
                    {
                        "tag": "#text",
                        "text": f"A interface completa foi bloqueada pelo site ({reason}); exibindo conteúdo extraído.",
                    }
                ],
            },
        ]
        for paragraph in paragraphs:
            children.append(
                {
                    "tag": "p",
                    "attributes": {},
                    "style": {"font-size": "16px", "line-height": "24px"},
                    "children": [{"tag": "#text", "text": paragraph[:MAX_TEXT_CHARS]}],
                }
            )
        return {
            "title": str(item.get("title") or "Página"),
            "url": str(item.get("url") or url),
            "favicon": str(item.get("favicon") or ""),
            "viewport": {"width": 1280, "height": 720},
            "nodeCount": len(children),
            "truncated": len(paragraphs) >= 120,
            "mode": "fallback_text",
            "body": {
                "tag": "body",
                "attributes": {},
                "style": {
                    "display": "block",
                    "background-color": "rgb(255, 255, 255)",
                    "color": "rgb(32, 31, 36)",
                },
                "children": children,
            },
        }
    except (httpx.HTTPError, ValueError, TypeError):
        return None


async def _render_with_browser(runtime: Runtime, url: str, viewport: Viewport) -> Dict[str, Any]:
    browser = await _ensure_browser(runtime)
    context: Optional[BrowserContext] = None

    async with runtime.semaphore:
        try:
            context = await browser.new_context(
                viewport={"width": viewport.width, "height": viewport.height},
                locale="pt-BR",
                java_script_enabled=True,
                ignore_https_errors=False,
                accept_downloads=False,
                service_workers="block",
                reduced_motion="reduce",
            )
            context.set_default_timeout(ACTION_TIMEOUT_MS)
            context.set_default_navigation_timeout(NAV_TIMEOUT_MS)
            await context.route("**/*", lambda route, req: _route_guard(runtime, route, req))
            if not ALLOW_WEBSOCKETS:
                # Routed WebSockets do not connect to the real server unless explicitly
                # connected; closing them avoids an SSRF/long-lived-connection bypass.
                await context.route_web_socket("**/*", lambda ws: ws.close(code=1000, reason="blocked"))

            page = await context.new_page()

            response = await page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=NAV_TIMEOUT_MS,
            )

            origin_status = response.status if response is not None else None
            if origin_status is not None and origin_status >= 400:
                fallback = await _tavily_extract_fallback(
                    runtime, url, f"HTTP {origin_status}"
                )
                if fallback is not None:
                    return fallback
                return {
                    "_error": True,
                    "error": f"O site respondeu HTTP {origin_status}.",
                    "originStatus": origin_status,
                }

            if RENDER_SETTLE_MS:
                await page.wait_for_timeout(RENDER_SETTLE_MS)

            # Give already-started font loading a brief chance without waiting for network-idle.
            try:
                await page.evaluate(
                    "() => Promise.race([document.fonts ? document.fonts.ready : Promise.resolve(), new Promise(r => setTimeout(r, 500))])"
                )
            except PlaywrightError:
                pass

            document = await page.evaluate(
                SERIALIZE_DOM_JS,
                {
                    "maxNodes": MAX_NODES,
                    "maxDepth": MAX_DEPTH,
                    "maxText": MAX_TEXT_CHARS,
                },
            )
            if not isinstance(document, dict) or not document.get("body"):
                return {"_error": True, "error": "A página não gerou um DOM renderizável."}

            document["mode"] = "dom"
            document = _truncate_tree_to_size(document, MAX_RESPONSE_BYTES)
            return document

        except PlaywrightTimeoutError:
            fallback = await _tavily_extract_fallback(runtime, url, "timeout")
            if fallback is not None:
                return fallback
            return {
                "_error": True,
                "error": "O site demorou demais para carregar.",
                "timeout": True,
            }
        except PlaywrightError as exc:
            fallback = await _tavily_extract_fallback(runtime, url, "bloqueio de navegação")
            if fallback is not None:
                return fallback
            return {
                "_error": True,
                "error": f"Falha ao renderizar a página: {str(exc)[:300]}",
            }
        finally:
            if context is not None:
                try:
                    await context.close()
                except Exception:
                    pass


@app.post("/search", dependencies=[Depends(require_api_key)])
async def search(payload: SearchRequestModel, request: Request) -> JSONResponse:
    runtime = _runtime(request)
    if not TAVILY_API_KEY:
        return JSONResponse(
            {
                "success": False,
                "error": "TAVILY_API_KEY não configurada no servidor.",
                "type": "search",
            },
            status_code=200,
        )

    cache_key = f"{payload.max_results}|{payload.query.casefold()}"
    cached = await runtime.search_cache.get(cache_key)
    if cached is not None:
        return JSONResponse(cached)

    try:
        response = await runtime.http.post(
            "https://api.tavily.com/search",
            headers={
                "Authorization": f"Bearer {TAVILY_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "query": payload.query,
                "search_depth": "basic",
                "max_results": payload.max_results,
                "include_answer": False,
                "include_images": False,
                "include_raw_content": False,
            },
        )
    except httpx.TimeoutException:
        return JSONResponse(
            {"success": False, "error": "Tavily demorou demais para responder.", "type": "search"}
        )
    except httpx.HTTPError:
        return JSONResponse(
            {"success": False, "error": "Falha de rede ao acessar a Tavily.", "type": "search"}
        )

    if response.status_code != 200:
        try:
            detail = response.json().get("detail", {})
            message = detail.get("error") if isinstance(detail, dict) else None
        except ValueError:
            message = None
        if response.status_code == 429:
            message = message or "Tavily aplicou rate limit. Aguarde antes de pesquisar novamente."
        else:
            message = message or f"Tavily respondeu HTTP {response.status_code}."
        return JSONResponse({"success": False, "error": message, "type": "search"})

    try:
        data = response.json()
    except ValueError:
        return JSONResponse(
            {"success": False, "error": "Tavily retornou JSON inválido.", "type": "search"}
        )

    clean_results = []
    for item in data.get("results") or []:
        if not isinstance(item, dict):
            continue
        clean_results.append(
            {
                "title": str(item.get("title") or "")[:200],
                "url": str(item.get("url") or "")[:2048],
                "content": str(item.get("content") or "")[:1600],
                "score": float(item.get("score") or 0.0),
            }
        )

    result = {
        "success": True,
        "type": "search",
        "query": payload.query,
        "results": clean_results,
    }
    await runtime.search_cache.set(cache_key, result, SEARCH_CACHE_TTL)
    return JSONResponse(result)


@app.post("/render", dependencies=[Depends(require_api_key)])
async def render(payload: RenderRequestModel, request: Request) -> JSONResponse:
    runtime = _runtime(request)

    try:
        url = _normalize_url(payload.url)
        await _assert_public_url(runtime, url)
    except ValueError as exc:
        # Application-level navigation errors are returned as HTTP 200 so the current
        # Roblox BrowseServer can decode and show the real message instead of collapsing
        # it into a generic 4xx/5xx error.
        return JSONResponse({"success": False, "type": "document", "error": str(exc)})

    cache_key = f"{payload.viewport.width}x{payload.viewport.height}|{url}"
    cached = await runtime.render_cache.get(cache_key)
    if cached is not None:
        return JSONResponse(cached)

    document = await _render_with_browser(runtime, url, payload.viewport)
    if document.pop("_error", False):
        return JSONResponse(
            {
                "success": False,
                "type": "document",
                **document,
            }
        )

    result = {
        "success": True,
        "type": "document",
        "document": document,
    }
    await runtime.render_cache.set(cache_key, result, RENDER_CACHE_TTL)
    return JSONResponse(result)


if __name__ == "__main__":
    import uvicorn

    # One worker is intentional: each worker owns a full browser process. Scale with
    # multiple containers/processes only when you understand the memory implications.
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=False,
        workers=1,
        proxy_headers=False,
    )
