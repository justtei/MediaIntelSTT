"""WebVTT caption export — the one Phase 4 broadcast-integration item
(requirements.md: "CG system output integration (WebVTT/SCC)") that's
achievable without vendor character-generator hardware or a live broadcast
feed to test against. transcribe.py writes a .vtt alongside the plain
transcript whenever the backend returns timestamped chunks.
"""


def _ts(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:06.3f}"


def format_vtt(chunks) -> str:
    """chunks: iterable of objects with .start (s), .end (s), .text — e.g.
    backends.Segment. Empty-text chunks are skipped."""
    lines = ["WEBVTT", ""]
    for c in chunks:
        if not c.text:
            continue
        lines.append(f"{_ts(c.start)} --> {_ts(c.end)}")
        lines.append(c.text)
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"
