from url_safety import validate_stream_url

ALLOWED = frozenset({"youtube.com", "dailymotion.com"})


def test_allowed_host_passes():
    assert validate_stream_url("https://youtube.com/watch?v=x", ALLOWED) is None


def test_allowed_subdomain_passes():
    assert validate_stream_url("https://m.youtube.com/watch?v=x", ALLOWED) is None


def test_unlisted_host_rejected():
    reason = validate_stream_url("https://evil.example.com/", ALLOWED)
    assert reason is not None and "allow-list" in reason


def test_non_http_scheme_rejected():
    assert validate_stream_url("file:///etc/passwd", ALLOWED) is not None
    assert validate_stream_url("ftp://youtube.com/x", ALLOWED) is not None


def test_ip_literal_rejected_even_if_would_be_allowed():
    assert validate_stream_url("http://127.0.0.1/admin", ALLOWED) is not None
    assert validate_stream_url("http://169.254.169.254/latest/meta-data/", ALLOWED) is not None


def test_ipv6_literal_rejected():
    assert validate_stream_url("http://[::1]/", ALLOWED) is not None


def test_empty_url_rejected():
    assert validate_stream_url("", ALLOWED) is not None
    assert validate_stream_url(None, ALLOWED) is not None


def test_no_host_rejected():
    assert validate_stream_url("https:///path", ALLOWED) is not None


def test_allow_any_host_bypasses_allowlist_but_not_scheme_or_ip_checks():
    assert validate_stream_url("https://evil.example.com/", ALLOWED, allow_any_host=True) is None
    assert validate_stream_url("http://127.0.0.1/", ALLOWED, allow_any_host=True) is not None
    assert validate_stream_url("file:///etc/passwd", ALLOWED, allow_any_host=True) is not None
