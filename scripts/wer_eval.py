"""Normalized WER/CER for Urdu: strips punctuation and diacritics, folds
Arabic-drift codepoints into Urdu forms, then scores with jiwer.

Usage: python scripts/wer_eval.py refs/clip.txt transcripts/clip.openvino.txt
"""
import argparse
import sys
from pathlib import Path

import jiwer

from urdu_norm import normalize


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
    ref, hyp = normalize(ref_raw), normalize(hyp_raw)

    out = jiwer.process_words(ref, hyp)
    print(f"normalized WER: {out.wer:.1%}   "
          f"(sub {out.substitutions}, del {out.deletions}, ins {out.insertions} "
          f"over {out.substitutions + out.deletions + out.hits} ref words)")
    print(f"normalized CER: {jiwer.cer(ref, hyp):.1%}")
    if args.raw:
        print(f"raw WER (no normalization): {jiwer.wer(ref_raw, hyp_raw):.1%}")


if __name__ == "__main__":
    main()
