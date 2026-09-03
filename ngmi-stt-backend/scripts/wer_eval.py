"""Normalized WER/CER for Urdu: strips punctuation and diacritics, folds
Arabic-drift codepoints into Urdu forms, then scores with jiwer.

Usage: python scripts/wer_eval.py refs/clip.txt transcripts/clip.openvino.txt
"""
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import jiwer

from urdu_norm import normalize


@dataclass
class ScoreResult:
    wer: float
    cer: float
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    raw_wer: float | None = None


def score(ref_raw: str, hyp_raw: str, include_raw: bool = False) -> ScoreResult:
    """Normalize both transcripts and compute WER/CER. Pure function, no I/O."""
    ref, hyp = normalize(ref_raw), normalize(hyp_raw)
    out = jiwer.process_words(ref, hyp)
    return ScoreResult(
        wer=out.wer, cer=jiwer.cer(ref, hyp),
        substitutions=out.substitutions, deletions=out.deletions,
        insertions=out.insertions, hits=out.hits,
        raw_wer=jiwer.wer(ref_raw, hyp_raw) if include_raw else None,
    )


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("reference", help="hand-corrected reference transcript")
    ap.add_argument("hypothesis", help="model output transcript")
    ap.add_argument("--raw", action="store_true", help="also show unnormalized WER")
    args = ap.parse_args()

    ref_raw = Path(args.reference).read_text(encoding="utf-8")
    hyp_raw = Path(args.hypothesis).read_text(encoding="utf-8")
    result = score(ref_raw, hyp_raw, include_raw=args.raw)

    print(f"normalized WER: {result.wer:.1%}   "
          f"(sub {result.substitutions}, del {result.deletions}, ins {result.insertions} "
          f"over {result.substitutions + result.deletions + result.hits} ref words)")
    print(f"normalized CER: {result.cer:.1%}")
    if args.raw:
        print(f"raw WER (no normalization): {result.raw_wer:.1%}")


if __name__ == "__main__":
    main()
