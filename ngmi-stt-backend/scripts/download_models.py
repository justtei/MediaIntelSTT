"""Download pre-converted OpenVINO Whisper models from HuggingFace.

Usage:
  python scripts/download_models.py                 # large-v3 int8 (default)
  python scripts/download_models.py small-int8      # quick smoke-test model
"""
import sys
from pathlib import Path

import certifi
import httpx
from huggingface_hub.utils import set_client_factory, set_async_client_factory
from huggingface_hub import snapshot_download

set_client_factory(lambda: httpx.Client(verify=certifi.where()))
set_async_client_factory(lambda: httpx.AsyncClient(verify=certifi.where()))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHOICES = {
    "large-v3-int8": "OpenVINO/whisper-large-v3-int8-ov",
    "large-v3-int4": "OpenVINO/whisper-large-v3-int4-ov",
    "large-v3-turbo-int8": "OpenVINO/whisper-large-v3-turbo-int8-ov",
    "medium-int8": "OpenVINO/whisper-medium-int8-ov",
    "small-int8": "OpenVINO/whisper-small-int8-ov",
}


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "medium-int8"
    if key not in CHOICES:
        sys.exit(f"unknown model '{key}'; choices: {', '.join(CHOICES)}")
    repo = CHOICES[key]
    target = PROJECT_ROOT / "models" / repo.split("/")[1]
    print(f"downloading {repo} -> {target}")
    snapshot_download(repo_id=repo, local_dir=target)
    print("done")


if __name__ == "__main__":
    main()
