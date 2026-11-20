"""Publish onomatron artifacts to the Hugging Face Hub.

Two stages (see scripts/publish_hf.py):

- **export** (local, no network): read the MLflow/Optuna state persisted
  under ``data/benchmark/`` by the benchmark, select the top-N models by
  **val NLL** (the test split is never used for selection), and assemble
  under ``data/hf/``: one self-contained bundle + model card per model, the
  dataset + its card, and the benchmark results + tracking DBs;
- **push**: upload the prepared tree to HF repos with huggingface_hub
  (dataset / model), token from ``HF_TOKEN`` or `hf auth login`.

Repo layout for namespace ``<ns>``:
``<ns>/onomatron-names`` (dataset), ``<ns>/onomatron-benchmark`` (results +
mlflow.db + optuna.db), ``<ns>/onomatron-<model>`` per top model.
"""

import json
import logging
from pathlib import Path
from typing import Any

from onomatron.benchmark.config import BenchmarkConfig
from onomatron.benchmark.dataset import MODEL_CLASSES
from onomatron.data import Vocab, load_names
from onomatron.publish import cards, export, push

logger = logging.getLogger(__name__)

__all__ = ["run"]


def _repo(namespace: str | None, name: str) -> str:
    return f"{namespace}/{name}" if namespace else f"<namespace>/{name}"


def run(
    *,
    namespace: str | None = None,
    top_n: int = 3,
    mode: str = "export",
    out_root: str | Path = "data/hf",
    models: list[str] | None = None,
    cfg: BenchmarkConfig | None = None,
) -> dict[str, Any]:
    """Export (and optionally push) the publishable artifacts.

    mode="export" is local-only; mode="push" additionally uploads to HF and
    requires `namespace` + a token (HF_TOKEN or `hf auth login`).
    Returns a small summary dict.
    """
    if mode not in ("export", "push"):
        raise ValueError(f"unknown mode {mode!r}")
    if mode == "push" and not namespace:
        raise ValueError("mode='push' requires a namespace (HF user or org)")
    cfg = cfg or BenchmarkConfig()
    out = Path(out_root)
    runs = export.load_eval_runs(cfg)
    top = export.select_top_n(runs, top_n, models)
    if not top:
        raise RuntimeError("no eval run to publish — run the benchmark first")
    logger.info("top %d by val NLL: %s", len(top), [r.model for r in top])

    vocab = Vocab(load_names(cfg.names_path))
    results_md = cards.results_table_md(runs, highlight={r.model for r in top})
    dataset_repo = _repo(namespace, push.DATASET_REPO)
    benchmark_repo = _repo(namespace, push.BENCHMARK_REPO)

    bundles: dict[str, Path] = {}
    for r in top:
        bundle = export.export_model_bundle(r, cfg, out)
        cls_name = (
            "NGramModel" if r.model == "ngram" else MODEL_CLASSES[r.model].__name__
        )
        cards.make_model_card(
            r,
            model_repo=_repo(namespace, f"onomatron-{r.model}"),
            dataset_repo=dataset_repo,
            benchmark_repo=benchmark_repo,
            results_md=results_md,
            samples=export.read_samples(r),
            cls_name=cls_name,
            vocab_size=len(vocab),
        ).save(bundle / "README.md")
        bundles[r.model] = bundle

    bench_dir = export.export_results(runs, cfg, out / "benchmark")
    (bench_dir / "results.md").write_text(
        f"# onomatron benchmark results\n\n{results_md}\n", encoding="utf-8"
    )

    stats = export.dataset_stats(cfg)
    ds_dir = export.export_dataset(cfg, out / "dataset")
    cards.make_dataset_card(
        stats, dataset_repo=dataset_repo, benchmark_repo=benchmark_repo
    ).save(ds_dir / "README.md")

    summary: dict[str, Any] = {
        "out_root": str(out),
        "n_runs": len(runs),
        "top": [r.model for r in top],
        "bundles": {m: str(p) for m, p in bundles.items()},
        "dataset_dir": str(ds_dir),
        "benchmark_dir": str(bench_dir),
        "pushed": [],
    }
    if mode == "push":
        api = push.get_api()
        summary["pushed"] = [
            push.publish_dataset(api, namespace, ds_dir),
            push.publish_benchmark(api, namespace, bench_dir),
            *(push.publish_model(api, namespace, m, p) for m, p in bundles.items()),
        ]
        logger.info("pushed: %s", summary["pushed"])
    (out / "manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary
