from dataclasses import dataclass

from webvtt import format_vtt, _ts


@dataclass
class Chunk:
    start: float
    end: float
    text: str


def test_timestamp_formatting():
    assert _ts(0) == "00:00:00.000"
    assert _ts(2.3) == "00:00:02.300"
    assert _ts(65.5) == "00:01:05.500"
    assert _ts(3661.25) == "01:01:01.250"


def test_negative_clamped_to_zero():
    assert _ts(-1.0) == "00:00:00.000"


def test_format_vtt_basic():
    out = format_vtt([Chunk(0.0, 1.5, "hello"), Chunk(1.5, 3.0, "world")])
    assert out.startswith("WEBVTT\n\n")
    assert "00:00:00.000 --> 00:00:01.500" in out
    assert "hello" in out
    assert "00:00:01.500 --> 00:00:03.000" in out
    assert "world" in out


def test_empty_text_chunks_skipped():
    out = format_vtt([Chunk(0.0, 1.0, ""), Chunk(1.0, 2.0, "spoken")])
    assert "spoken" in out
    assert out.count("-->") == 1


def test_empty_chunks_still_has_header():
    assert format_vtt([]) == "WEBVTT\n"
