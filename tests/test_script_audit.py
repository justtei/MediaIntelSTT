import io
from contextlib import redirect_stdout

from script_audit import audit

URDU_TEXT = "یہ ایک بہترین خبر ہے، اور گاؤں کی سڑک پر لوگ کھڑے ہیں۔"
ENGLISH_TEXT = "This is a real, correct English news transcript about the weather today."
ROMANIZED_URDU = "yeh aik behtareen khabar hai"


def run(text, language="ur"):
    with redirect_stdout(io.StringIO()):
        return audit(text, language)


def test_correct_urdu_passes_ur():
    assert run(URDU_TEXT, "ur") == 0


def test_romanized_urdu_flagged_suspect():
    assert run(ROMANIZED_URDU, "ur") == 1


def test_correct_english_false_positive_under_ur_language():
    # Documents issue #4: without --language en, correct English is misflagged.
    assert run(ENGLISH_TEXT, "ur") == 1


def test_correct_english_passes_under_en_language():
    assert run(ENGLISH_TEXT, "en") == 0


def test_auto_detects_urdu():
    assert run(URDU_TEXT, "auto") == 0


def test_auto_detects_english():
    assert run(ENGLISH_TEXT, "auto") == 0


def test_empty_text_fails():
    assert run("", "ur") == 1
