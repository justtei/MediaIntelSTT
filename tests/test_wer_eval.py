from wer_eval import score


def test_identical_transcripts_score_zero():
    text = "یہ ایک ٹیسٹ ہے۔"
    result = score(text, text)
    assert result.wer == 0.0
    assert result.cer == 0.0


def test_deletion_counted_and_wer_computed():
    ref = "یہ ایک ٹیسٹ ہے۔"
    hyp = "یہ ٹیسٹ ہے"
    result = score(ref, hyp, include_raw=True)
    assert result.deletions == 1
    assert result.substitutions == 0
    assert result.insertions == 0
    assert round(result.wer, 2) == 0.25
    assert result.raw_wer is not None


def test_raw_score_omitted_by_default():
    result = score("a", "a")
    assert result.raw_wer is None


def test_normalization_hides_diacritic_only_differences():
    # Diacritics and Arabic-drift codepoints are noise, not real errors.
    ref = "مدرسہ"
    hyp = "مَدرسة"  # same word with a diacritic + Arabic-drift teh marbuta
    result = score(ref, hyp)
    assert result.wer == 0.0
