from urdu_norm import normalize


def test_arabic_drift_folds_to_urdu_forms():
    assert normalize("يك") == "یک"  # ARABIC YEH+KAF -> FARSI YEH, KEHEH
    assert normalize("مدرسة") == normalize("مدرسہ")  # TEH MARBUTA -> HEH GOAL


def test_diacritics_stripped():
    assert normalize("مَدرسہ") == "مدرسہ"


def test_punctuation_replaced_with_space_and_collapsed():
    assert normalize("سلام، دنیا۔") == "سلام دنیا"


def test_whitespace_collapsed_and_stripped():
    assert normalize("  multiple   spaces  here  ") == "multiple spaces here"


def test_ascii_punctuation_stripped():
    assert normalize("Hello, World!") == "Hello World"


def test_idempotent():
    text = "یہ ایک ٹیسٹ ہے۔"
    once = normalize(text)
    assert normalize(once) == once


def test_empty_string():
    assert normalize("") == ""
