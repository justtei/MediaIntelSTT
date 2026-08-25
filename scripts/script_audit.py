"""Audit a transcript's script before trusting any WER number.

Catches the two silent failure modes for Urdu ASR output:
  1. romanization (Latin chars where Urdu should be)
  2. Arabic drift (ARABIC YEH/KAF/TEH MARBUTA instead of Urdu forms)

Usage: python scripts/script_audit.py transcripts/clip.openvino.txt
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

ARABIC_BLOCKS = (
    (0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF), (0xFE70, 0xFEFF),
)
# Letters Urdu orthography prefers over their generic-Arabic counterparts.
URDU_MARKERS = set("ٹڈڑںھہۃکگیے")
ARABIC_DRIFT = {"ي": "ARABIC YEH (want ی)", "ك": "ARABIC KAF (want ک)",
                "ة": "TEH MARBUTA (want ہ)", "ى": "ALEF MAKSURA (want ی)"}


def in_arabic_block(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in ARABIC_BLOCKS)


def audit(text: str) -> int:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        print("no letters found — empty or non-text output"); return 1
    n = len(letters)
    arabic = sum(1 for c in letters if in_arabic_block(c))
    latin = sum(1 for c in letters if "a" <= c.lower() <= "z")
    other = n - arabic - latin
    markers = Counter(c for c in letters if c in URDU_MARKERS)
    drift = Counter(c for c in letters if c in ARABIC_DRIFT)

    print(f"letters: {n}")
    print(f"  Perso-Arabic script: {arabic/n:6.1%}")
    print(f"  Latin (romanized?):  {latin/n:6.1%}")
    print(f"  other:               {other/n:6.1%}")
    print(f"Urdu-specific letters seen: {sum(markers.values())} "
          f"({', '.join(sorted(markers)) if markers else 'NONE'})")
    if drift:
        print("Arabic-drift characters (normalize before WER, or the model is drifting):")
        for ch, cnt in drift.most_common():
            print(f"  {ch}  x{cnt}  {ARABIC_DRIFT[ch]}")

    ok = arabic / n >= 0.9 and markers
    print("\nVERDICT:", "OK — output is Urdu script" if ok else
          "SUSPECT — check for romanization or wrong-language decode")
    return 0 if ok else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript")
    args = ap.parse_args()
    sys.exit(audit(Path(args.transcript).read_text(encoding="utf-8")))
