"""Export benchmark state into publishable artifacts (local only, no network).

Reads the MLflow eval runs + Optuna studies persisted under ``data/benchmark/``
by ``scripts/benchmark.py`` and assembles, under ``data/hf/``:

- ``models/<model>/``: one self-contained bundle (config.json + model.pt +
  vendored onomatron subset + generate.py + requirements.txt) per top model;
- ``benchmark/``: results.csv, the MLflow/Optuna databases (audit trail);
- ``dataset/``: names.txt + dataset_stats.json (the README card is written by
  the caller, see onomatron.publish.cards).

Model selection ranks by val NLL only — the test split is reported once, in
the cards, never used for selection.
"""

import ast
import csv
import json
import logging
import math
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlflow.tracking import MlflowClient

from onomatron.benchmark.config import BenchmarkConfig
from onomatron.benchmark.dataset import MODEL_CLASSES
from onomatron.benchmark.evaluate import EVAL_EXPERIMENT
from onomatron.benchmark.spaces import TRAIN_KEYS
from onomatron.data import Vocab, load_names, split_names
from onomatron.models.ngram import NGramModel
from onomatron.scraper.sources import SOURCES
from onomatron.serialize import (
    WEIGHTS_NAME,
    make_config,
    save_bundle,
    write_bundle,
)

logger = logging.getLogger(__name__)

# Real template files (generate.py, requirements.txt, onomatron_init.py, *.md.j2)
# live at the repo root under templates/ — copied/rendered verbatim at publish.
TEMPLATES_DIR = Path(__file__).resolve().parents[3] / "templates"

# Params logged by the eval runs alongside the architecture ones (evaluate.py
# logs arch + training + run metadata in one dict) — all excluded from the
# bundle config's constructor params.
META_PARAMS: tuple[str, ...] = (
    "model",
    "seed",
    "final_steps",
    "patience",
    "eval_every",
)

# Metrics copied into each bundle's config.json as provenance.
BUNDLE_METRICS: tuple[str, ...] = (
    "val_nll",
    "test_nll",
    "n_parameters",
    "model_size_mb",
    "train_wall_time_s",
    "train_peak_mem_mb",
    "examples_per_sec",
    "inference_names_per_sec",
    "inference_chars_per_sec",
    "best_step",
)

# Columns of results.csv, in order.
RESULT_METRICS: tuple[str, ...] = (
    "val_nll",
    "test_nll",
    "n_parameters",
    "model_size_mb",
    "train_wall_time_s",
    "inference_names_per_sec",
)

# License / attribution per source kind, shown in the dataset card.
KIND_LICENSES: dict[str, tuple[str, str, str]] = {
    "list_page": (
        "French Wikipedia",
        "CC BY-SA 4.0",
        "https://creativecommons.org/licenses/by-sa/4.0/",
    ),
    "category": (
        "French Wikipedia",
        "CC BY-SA 4.0",
        "https://creativecommons.org/licenses/by-sa/4.0/",
    ),
    "work_page": (
        "French Wikipedia",
        "CC BY-SA 4.0",
        "https://creativecommons.org/licenses/by-sa/4.0/",
    ),
    "sparql": (
        "Wikidata (WDQS)",
        "CC0 1.0",
        "https://creativecommons.org/publicdomain/zero/1.0/",
    ),
    "hf_dataset": (
        "ObTIC — compnet-renard/7-romans-alias-resolution",
        "no license declared",
        "cite Maurel et al., DH 2025 (see card)",
    ),
    "raw_file": (
        "local file data/external/*.txt (gitignored)",
        "not declared",
        "local-only input, not committed",
    ),
}


@dataclass
class EvalRun:
    """One final-evaluation MLflow run (experiment "onomatron-benchmark")."""

    model: str
    run_id: str
    run_name: str
    params: dict[str, str]  # MLflow stores params as strings
    metrics: dict[str, float]
    tags: dict[str, str]
    artifact_dir: Path


def _coerce(value: str) -> Any:
    """Parse a MLflow param string back to int/float/bool/None/list when
    possible (they were logged via str()); leave real strings untouched."""
    try:
        return ast.literal_eval(value)
    except ValueError, SyntaxError:
        return value


def load_eval_runs(cfg: BenchmarkConfig) -> list[EvalRun]:
    """Load every eval run, ordered by val NLL ascending (best first).

    Raises RuntimeError if the benchmark has never run: there is nothing to
    publish without eval runs.
    """
    client = MlflowClient(tracking_uri=cfg.mlflow_tracking_uri)
    experiment = client.get_experiment_by_name(EVAL_EXPERIMENT)
    if experiment is None:
        raise RuntimeError(
            f"no MLflow experiment {EVAL_EXPERIMENT!r} in "
            f"{cfg.mlflow_tracking_uri} — "
            "run `uv run python scripts/benchmark.py --mode eval` first"
        )
    runs = client.search_runs(
        [experiment.experiment_id], order_by=["metrics.val_nll ASC"]
    )
    eval_runs = [
        EvalRun(
            model=r.data.tags.get("model") or str(r.data.params.get("model")),
            run_id=r.info.run_id,
            run_name=r.info.run_name,
            params=dict(r.data.params),
            metrics=dict(r.data.metrics),
            tags=dict(r.data.tags),
            artifact_dir=Path(r.info.artifact_uri.removeprefix("file://")),
        )
        for r in runs
    ]
    logger.info("loaded %d eval runs (best val_nll first)", len(eval_runs))
    return eval_runs


def arch_params(run: EvalRun) -> dict[str, Any]:
    """Constructor params of the model: logged params minus training keys
    (TRAIN_KEYS) and run metadata, with MLflow's string encoding undone."""
    skip = set(TRAIN_KEYS) | set(META_PARAMS)
    return {k: _coerce(v) for k, v in run.params.items() if k not in skip}


def select_top_n(
    runs: list[EvalRun], n: int, models: list[str] | None = None
) -> list[EvalRun]:
    """The n best runs by val NLL (runs must already be sorted), optionally
    restricted to `models`. Selection never looks at test_nll, and runs with
    a non-finite val NLL (diverged training, e.g. NaN on some devices) are
    ineligible — a NaN must never win a benchmark."""
    eligible = [
        r
        for r in runs
        if (models is None or r.model in models)
        and math.isfinite(r.metrics.get("val_nll", math.inf))
    ]
    for r in runs:
        if not math.isfinite(r.metrics.get("val_nll", math.inf)):
            logger.warning(
                "run %s has non-finite val_nll=%s — ineligible for top-N",
                r.model,
                r.metrics.get("val_nll"),
            )
    return eligible[:n]


def vendor_onomatron(dest: str | Path, model_names: list[str]) -> Path:
    """Copy the minimal onomatron subset able to load/sample `model_names`
    into ``dest/onomatron/`` — torch-only: no plotly, optuna, mlflow or
    scraper deps, so published model repos stay light."""
    import onomatron

    src = Path(onomatron.__file__).resolve().parent
    pkg = Path(dest) / "onomatron"
    (pkg / "models").mkdir(parents=True, exist_ok=True)
    shutil.copy(TEMPLATES_DIR / "onomatron_init.py", pkg / "__init__.py")
    shutil.copy(src / "data.py", pkg / "data.py")
    shutil.copy(src / "serialize.py", pkg / "serialize.py")
    shutil.copy(src / "models" / "__init__.py", pkg / "models" / "__init__.py")
    for name in model_names:
        shutil.copy(src / "models" / f"{name}.py", pkg / "models" / f"{name}.py")
    logger.info("vendored onomatron subset for %s into %s", model_names, pkg)
    return pkg


def export_model_bundle(
    run: EvalRun, cfg: BenchmarkConfig, out_root: str | Path
) -> Path:
    """Assemble the self-contained bundle of one eval run under
    ``out_root/models/<model>/`` (config.json + weights + vendored code +
    generate.py + requirements.txt). The README card is added by the caller."""
    names = load_names(cfg.names_path)
    vocab = Vocab(names)
    bundle = Path(out_root) / "models" / run.model
    params = arch_params(run)
    seed = int(_coerce(run.params.get("seed", cfg.seed)))
    metrics = {k: run.metrics[k] for k in BUNDLE_METRICS if k in run.metrics}
    if run.model == "ngram":
        # the eval run logs no checkpoint for the count-based model: refit it
        # deterministically from the same seeded train split
        names_train, _, _ = split_names(names, cfg.split_ratios, cfg.seed)
        net = NGramModel.fit(
            names_train, vocab, n=params["n"], smoothing=params["smoothing"]
        )
        save_bundle(
            bundle,
            model=run.model,
            net=net,
            params=params,
            vocab=vocab,
            seed=seed,
            metrics=metrics,
        )
    else:
        weights = run.artifact_dir / WEIGHTS_NAME
        if not weights.is_file():
            raise FileNotFoundError(
                f"no {WEIGHTS_NAME} in the artifacts of run {run.run_id} "
                f"({run.artifact_dir})"
            )
        config = make_config(
            run.model,
            MODEL_CLASSES[run.model].__name__,
            params,
            vocab,
            seed,
            metrics=metrics,
        )
        write_bundle(bundle, config=config, weights_path=weights)
    vendor_onomatron(bundle, [run.model])
    shutil.copy(TEMPLATES_DIR / "generate.py", bundle / "generate.py")
    shutil.copy(TEMPLATES_DIR / "requirements.txt", bundle / "requirements.txt")
    logger.info("exported bundle for %s -> %s", run.model, bundle)
    return bundle


def read_samples(run: EvalRun, limit: int = 20) -> list[str]:
    """The first `limit` sampled names logged as a run artifact."""
    path = run.artifact_dir / "samples.txt"
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8").splitlines()[:limit]


def export_results(
    runs: list[EvalRun], cfg: BenchmarkConfig, out_dir: str | Path
) -> Path:
    """Write results.csv (one row per eval run, best val NLL first, with the
    full coerced param dict) and copy the MLflow/Optuna databases so the whole
    benchmark stays auditable from the published artifacts."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "model", *RESULT_METRICS, "params"])
        for rank, r in enumerate(runs, start=1):
            writer.writerow(
                [
                    rank,
                    r.model,
                    *(r.metrics.get(m, "") for m in RESULT_METRICS),
                    json.dumps(
                        {k: _coerce(v) for k, v in r.params.items()}, ensure_ascii=False
                    ),
                ]
            )
    for db in ("mlflow.db", "optuna.db"):
        src = cfg.data_dir / db
        if src.is_file():
            shutil.copy(src, out / db)
    logger.info("wrote results.csv + tracking DBs into %s", out)
    return out


def dataset_stats(cfg: BenchmarkConfig) -> dict[str, Any]:
    """Statistics + per-source provenance of the final dataset.

    Per-source counts come from ``data/interim/<spec.name>.jsonl`` (one file
    per SourceSpec, written by run_scraper.py) — counting lines there is exact
    and avoids parsing the free-form `source` strings of names.jsonl.
    """
    names = load_names(cfg.names_path)
    interim = cfg.names_path.parent.parent / "interim"
    per_kind: Counter[str] = Counter()
    per_source: list[dict[str, Any]] = []
    for spec in SOURCES:
        path = interim / f"{spec.name}.jsonl"
        count = sum(1 for _ in path.open(encoding="utf-8")) if path.is_file() else 0
        per_kind[spec.kind] += count
        per_source.append(
            {
                "source": spec.name,
                "kind": spec.kind,
                "target": spec.target,
                "count": count,
            }
        )
    per_source.sort(key=lambda d: d["count"], reverse=True)
    lengths = [len(n) for n in names]
    return {
        "n_names": len(names),
        "n_candidates": sum(per_kind.values()),
        "alphabet": sorted(set("".join(names))),
        "len_min": min(lengths),
        "len_max": max(lengths),
        "len_mean": sum(lengths) / len(lengths),
        "per_kind": dict(per_kind),
        "per_source": per_source,
        "kind_licenses": {k: list(v) for k, v in KIND_LICENSES.items()},
    }


def export_dataset(cfg: BenchmarkConfig, out_dir: str | Path) -> Path:
    """Copy names.txt + dataset_stats.json into ``out_dir`` (card added by the
    caller)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg.names_path, out / "names.txt")
    stats = dataset_stats(cfg)
    (out / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("exported dataset (%d names) into %s", stats["n_names"], out)
    return out
