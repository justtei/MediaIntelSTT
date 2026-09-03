"""Urdu text normalization shared by the WER eval and script audit."""
import re
import unicodedata

# Arabic-preferred codepoints that drift into Whisper's Urdu output.
# Keys/values as escapes to keep this file ASCII-safe.
ARABIC_TO_URDU = {
    "ي": "ی",  # ARABIC YEH -> FARSI YEH
    "ى": "ی",  # ALEF MAKSURA -> FARSI YEH
    "ك": "ک",  # ARABIC KAF -> KEHEH
    "ة": "ہ",  # TEH MARBUTA -> HEH GOAL
    "ه": "ہ",  # ARABIC HEH -> HEH GOAL
}

# Harakat, Quranic annotation marks, superscript alef — orthographic noise for WER.
_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")

# Urdu-Arabic punctuation + ASCII punctuation + typographic quotes/dashes.
_PUNCT = re.compile(
    r"[،؛؟۔٪-٭"
    r"!-/:-@\[-`{-~‘’“”…–—]"
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    for src, dst in ARABIC_TO_URDU.items():
        text = text.replace(src, dst)
    text = _DIACRITICS.sub("", text)
    text = _PUNCT.sub(" ", text)
    return " ".join(text.split()).strip()
