"""Push the prepared data/hf/ tree to Hugging Face repos (network stage).

Everything here goes through huggingface_hub's HfApi; the token comes from
the HF_TOKEN environment variable or the cached `hf auth login` token, and is
never logged or committed. Repos are created public and idempotent
(exist_ok=True), so re-pushing updates files in place.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

DATASET_REPO = "onomatron-names"
BENCHMARK_REPO = "onomatron-benchmark"


def get_api(token: str | None = None):
    """Build a HfApi from `token`, else HF_TOKEN, else the cached token of
    `hf auth login`; raise a clear error if none is found."""
    from huggingface_hub import HfApi, get_token

    token = token or os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise RuntimeError(
            "no HF token found: set HF_TOKEN (a write token from "
            "https://huggingface.co/settings/tokens) or run "
            "`uv run hf auth login` once"
        )
    return HfApi(token=token)


def _upload_tree(api, repo_id: str, repo_type: str, local_dir: str | Path) -> None:
    """Create the repo if needed, then upload every file of `local_dir`
    (paths preserved relative to it)."""
    api.create_repo(repo_id, repo_type=repo_type, exist_ok=True)
    root = Path(local_dir)
    n = 0
    for file in sorted(root.rglob("*")):
        if file.is_file():
            api.upload_file(
                path_or_fileobj=str(file),
                path_in_repo=str(file.relative_to(root)),
                repo_id=repo_id,
                repo_type=repo_type,
            )
            n += 1
    logger.info("pushed %d files to %s (%s)", n, repo_id, repo_type)


def publish_dataset(api, namespace: str, local_dir: str | Path) -> str:
    repo_id = f"{namespace}/{DATASET_REPO}"
    _upload_tree(api, repo_id, "dataset", local_dir)
    return repo_id


def publish_model(api, namespace: str, model: str, local_dir: str | Path) -> str:
    repo_id = f"{namespace}/onomatron-{model}"
    _upload_tree(api, repo_id, "model", local_dir)
    return repo_id


def publish_benchmark(api, namespace: str, local_dir: str | Path) -> str:
    """The benchmark repo is a model-type repo used as an artifact bundle:
    results.csv/md + the MLflow and Optuna databases (full audit trail)."""
    repo_id = f"{namespace}/{BENCHMARK_REPO}"
    _upload_tree(api, repo_id, "model", local_dir)
    return repo_id
