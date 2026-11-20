"""Push the prepared data/hf/ tree to Hugging Face repos (network stage).

Everything here goes through huggingface_hub's HfApi; the token comes from
the HF_TOKEN environment variable and is never logged or committed. Repos
are created public and idempotent (exist_ok=True), so re-pushing updates
files in place.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

DATASET_REPO = "onomatron-names"
BENCHMARK_REPO = "onomatron-benchmark"
SPACE_REPO = "onomatron-space"


def get_api(token: str | None = None):
    """Build a HfApi from `token` or HF_TOKEN; raise a clear error if unset."""
    from huggingface_hub import HfApi

    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "no HF token: set HF_TOKEN (a write-access token from "
            "https://huggingface.co/settings/tokens)"
        )
    return HfApi(token=token)


def _upload_tree(
    api,
    repo_id: str,
    repo_type: str,
    local_dir: str | Path,
    *,
    space_sdk: str | None = None,
) -> None:
    """Create the repo if needed, then upload every file of `local_dir`
    (paths preserved relative to it)."""
    api.create_repo(repo_id, repo_type=repo_type, exist_ok=True, space_sdk=space_sdk)
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


def publish_space(api, namespace: str, local_dir: str | Path, model_repo: str) -> str:
    """Push the Space and point its ONOMATRON_MODEL_REPO variable at the best
    published model (best effort — needs a token with space rights)."""
    repo_id = f"{namespace}/{SPACE_REPO}"
    _upload_tree(api, repo_id, "space", local_dir, space_sdk="gradio")
    try:
        api.set_space_variable(repo_id, "ONOMATRON_MODEL_REPO", model_repo)
        logger.info("set space variable ONOMATRON_MODEL_REPO=%s", model_repo)
    except Exception as e:  # variable is optional: the upload stands without it
        logger.warning(
            "could not set the Space variable ONOMATRON_MODEL_REPO (%s); "
            "set it manually in the Space settings",
            e,
        )
    return repo_id
