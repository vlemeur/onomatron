"""HF cards built with the official huggingface_hub API, rendered from Jinja2
templates (templates/*.md.j2 at the repo root — the standard way to template
markdown; huggingface_hub's own card templates are Jinja2 as well).

Metadata goes into `card.data` (ModelCardData / DatasetCardData / SpaceCardData)
and is rendered as the YAML frontmatter by the library itself; the markdown
body is rendered from the .j2 template into `card.text`. Makers return the
card object — callers persist it with `card.save(path)`. Generating the cards
from the MLflow export guarantees the published numbers are exactly the
tracked ones.
"""

from pathlib import Path
from typing import Any

import jinja2
from huggingface_hub import DatasetCard, ModelCard, SpaceCard

from onomatron.publish.export import KIND_LICENSES, EvalRun, _coerce

TEMPLATES_DIR = Path(__file__).resolve().parents[3] / "templates"

_ENV = jinja2.Environment(
    loader=jinja2.FileSystemLoader(TEMPLATES_DIR),
    autoescape=False,  # markdown, not HTML
    keep_trailing_newline=True,
    trim_blocks=True,
    lstrip_blocks=True,
)


def _card(cls: type, **metadata: Any) -> Any:
    """Empty card of class `cls` with the given metadata fields set on
    card.data (rendered as YAML frontmatter by huggingface_hub)."""
    card = cls("---\n\n---\n", ignore_metadata_errors=True)
    for key, value in metadata.items():
        setattr(card.data, key, value)
    return card


def _render(template_name: str, **context: Any) -> str:
    return _ENV.get_template(template_name).render(**context)


def _f(value: float | None, nd: int = 4) -> str:
    return f"{value:.{nd}f}" if isinstance(value, int | float) else "—"


def results_table_md(runs: list[EvalRun], highlight: set[str] | None = None) -> str:
    """Full benchmark table (all eval runs, best val NLL first); models in
    `highlight` are bolded."""
    highlight = highlight or set()
    cols = [
        "model",
        "val_nll",
        "test_nll",
        "n_parameters",
        "size(MB)",
        "wall(s)",
        "names/s",
    ]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in runs:
        name = f"**{r.model}**" if r.model in highlight else r.model
        row = (
            f"| {name} | {_f(r.metrics.get('val_nll'))}"
            f" | {_f(r.metrics.get('test_nll'))}"
            f" | {int(r.metrics.get('n_parameters', 0))}"
            f" | {_f(r.metrics.get('model_size_mb'), 2)}"
            f" | {_f(r.metrics.get('train_wall_time_s'), 1)}"
            f" | {_f(r.metrics.get('inference_names_per_sec'), 0)} |"
        )
        lines.append(row)
    return "\n".join(lines)


def _params_table(params: dict[str, str]) -> str:
    rows = [f"| `{k}` | `{_coerce(v)!r}` |" for k, v in sorted(params.items())]
    return "\n".join(["| param | value |", "|---|---|", *rows])


def make_model_card(
    run: EvalRun,
    *,
    model_repo: str,
    dataset_repo: str,
    benchmark_repo: str,
    results_md: str,
    samples: list[str],
    cls_name: str,
    vocab_size: int,
) -> ModelCard:
    """ModelCard of one published model (saved as the repo's README.md)."""
    card = _card(
        ModelCard,
        license="mit",
        language=["fr"],
        library_name="pytorch",
        pipeline_tag="text-generation",
        tags=[
            "text-generation",
            "character-level-lm",
            "makemore",
            "french-literature",
        ],
        datasets=[dataset_repo],
    )
    arch_params = {
        k: v
        for k, v in run.params.items()
        if k not in ("model", "seed", "final_steps", "patience", "eval_every")
    }
    card.text = _render(
        "model_card.md.j2",
        run=run,
        model_repo=model_repo,
        dataset_repo=dataset_repo,
        benchmark_repo=benchmark_repo,
        results_md=results_md,
        samples_block="\n".join(samples) if samples else "(no samples logged)",
        cls_name=cls_name,
        vocab_size=vocab_size,
        n_params=int(run.metrics.get("n_parameters", 0)),
        seed=_coerce(run.params.get("seed", 42)),
        device=run.tags.get("device", "?"),
        params_table=_params_table(arch_params),
        f=_f,
    )
    return card


def make_dataset_card(
    stats: dict[str, Any], *, dataset_repo: str, benchmark_repo: str
) -> DatasetCard:
    """DatasetCard of the names dataset (saved as the repo's README.md)."""
    card = _card(
        DatasetCard,
        license="other",
        language=["fr"],
        tags=["french-literature", "character-names", "text-generation"],
        task_categories=["text-generation"],
    )
    per_kind = stats["per_kind"]
    kind_rows = [
        {
            "kind": kind,
            "count": per_kind.get(kind, 0),
            "origin": origin,
            "license": lic,
            "note": note,
        }
        for kind, (origin, lic, note) in KIND_LICENSES.items()
    ]
    src_rows = [s for s in stats["per_source"] if s["count"] > 0]
    card.text = _render(
        "dataset_card.md.j2",
        stats=stats,
        alphabet="".join(stats["alphabet"]),
        dataset_repo=dataset_repo,
        benchmark_repo=benchmark_repo,
        kind_rows=kind_rows,
        src_rows=src_rows,
    )
    return card


def make_space_card(model_repo: str) -> SpaceCard:
    """SpaceCard of the Gradio demo (saved as the Space's README.md; the sdk
    and app_file metadata drive the Space build)."""
    card = _card(
        SpaceCard,
        title="onomatron",
        emoji="📖",
        colorFrom="blue",
        colorTo="purple",
        sdk="gradio",
        app_file="app.py",
        pinned=False,
        license="mit",
    )
    card.text = _render("space.md.j2", model_repo=model_repo)
    return card
