"""URL safety checks for StreamPuller (webapp/server.py issue #11).

Blocks the obvious SSRF vectors before a client-submitted URL is ever handed
to yt-dlp/ffmpeg: non-http(s) schemes, IP-literal hosts (127.0.0.1, cloud
metadata endpoints like 169.254.169.254, internal IPs), and hosts outside an
explicit allow-list.

Not a full SSRF-proof resolver — a domain that's allow-listed but DNS-rebound
to an internal IP at request time isn't caught by string matching alone. That
needs a network-layer control (e.g. a proxy that resolves-then-checks-then-
connects to a pinned IP). This is the practical app-level gate for a tool
whose server is expected to bind to localhost; harden further before binding
to 0.0.0.0 or a LAN-facing interface.
"""
import ipaddress
from urllib.parse import urlparse

DEFAULT_ALLOWED_HOSTS = frozenset({
    "youtube.com", "youtu.be", "m.youtube.com",
    "dailymotion.com",
    "vimeo.com",
    "twitch.tv",
})


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False


def validate_stream_url(url: str, allowed_hosts=DEFAULT_ALLOWED_HOSTS,
                         allow_any_host: bool = False) -> str | None:
    """Return None if `url` is safe to pull, else a human-readable rejection reason."""
    if not url or not isinstance(url, str):
        return "empty URL"
    try:
        parsed = urlparse(url)
    except ValueError:
        return "unparseable URL"
    if parsed.scheme not in ("http", "https"):
        return f"scheme {parsed.scheme!r} not allowed — only http/https"
    host = parsed.hostname
    if not host:
        return "URL has no host"
    if _is_ip_literal(host):
        return "IP-literal hosts are not allowed (SSRF risk) — use a domain name"
    if allow_any_host:
        return None
    host = host.lower()
    if host in allowed_hosts or any(host.endswith("." + h) for h in allowed_hosts):
        return None
    return f"host {host!r} is not in the stream allow-list"
