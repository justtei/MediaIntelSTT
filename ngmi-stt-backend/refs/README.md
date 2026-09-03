# Reference transcripts (ground truth)

Tracked in git — unlike `audio/`, `transcripts/`, and `models/`, this directory
is *not* gitignored. It's evidence, not a build artifact (see `requirements.md`
§7, §10).

## Why this directory was empty

`wer_eval.py` scores a model transcript against a **hand-corrected** reference.
That correction step requires a human listening to the source audio and fixing
whatever the model got wrong — it cannot be generated from the model's own
output (that would just measure the model against itself) or written without
hearing the clip.

## Workflow

1. Get a real clip: `python scripts/fetch_audio.py "<url>" --name <clip> --section "<start>-<end>"`
2. Transcribe it: `python scripts/transcribe.py audio/<clip>.wav` (writes
   `transcripts/<clip>.openvino.txt`)
3. Listen to `audio/<clip>.wav` and hand-correct a copy of that transcript —
   fix mis-heard words, wrong names, punctuation — save as `refs/<clip>.txt`
4. Score it: `python scripts/wer_eval.py refs/<clip>.txt transcripts/<clip>.openvino.txt --raw`

Cover studio/field/panel/code-switched conditions per `requirements.md` §6.1,
not just one easy clip — a single anchor-desk sample won't represent
real broadcast WER.
