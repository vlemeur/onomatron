"""Benchmark / hyperparameter tuning entry point (thin CLI).

All configuration defaults live in onomatron.benchmark.BenchmarkConfig — this
script only forwards a few CLI overrides and calls onomatron.benchmark.run().

    uv run python scripts/benchmark.py --mode all --models mlp wavenet
    uv run python scripts/benchmark.py --mode tune --trials 50 --device mps
    uv run python scripts/benchmark.py --mode eval          # uses saved studies

Results: Optuna studies in data/benchmark/optuna.db, MLflow runs in
data/benchmark/mlflow.db (+ artifacts/); inspect with `mlflow ui`.
"""

import argparse
import logging
import sys

import optuna
from tqdm import tqdm

from onomatron.benchmark import MODELS, run

logger = logging.getLogger(__name__)


class TqdmLogHandler(logging.StreamHandler):
    """Route log records through tqdm.write: messages are printed ABOVE the
    active progress bars (which are redrawn right after), instead of
    interleaving with bar frames and making the terminal scroll."""

    def emit(self, record: logging.LogRecord) -> None:
        tqdm.write(self.format(record), file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["tune", "eval", "all"],
        default="all",
        help="tune: Optuna only, eval: final benchmark only, all: both",
    )
    parser.add_argument(
        "--models", nargs="+", choices=list(MODELS), default=list(MODELS)
    )
    parser.add_argument("--trials", type=int, default=None, help="trials per model")
    parser.add_argument("--steps", type=int, default=None, help="max steps per trial")
    parser.add_argument(
        "--final-steps", type=int, default=None, help="steps of the final eval run"
    )
    parser.add_argument("--device", default=None, help="auto | cpu | mps | cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="keep per-step train bars and the data-loading INFO logs",
    )
    args = parser.parse_args()

    handler = TqdmLogHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    logging.root.handlers = [handler]
    logging.root.setLevel(logging.INFO)

    if not args.verbose:
        # Quiet the noisy per-trial logs: Optuna's own chatter, MLflow's
        # store messages, and the data/model INFO lines (they stay visible
        # with --verbose). The onomatron.benchmark milestone logs stay INFO.
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        for name in ("onomatron.data", "onomatron.models", "mlflow"):
            logging.getLogger(name).setLevel(logging.WARNING)

    overrides = {
        key: value
        for key, value in {
            "n_trials": args.trials,
            "steps": args.steps,
            "final_steps": args.final_steps,
            "device": args.device,
            "seed": args.seed,
            "verbose": args.verbose,
        }.items()
        if value is not None
    }
    results = run(mode=args.mode, models=args.models, **overrides)

    eval_results = results.get("eval")
    if eval_results:
        print("\nBenchmark summary (see MLflow for the full metric set):")
        width = max(len(m) for m in eval_results)
        cols = ("val_nll", "test_nll", "wall_s", "names/s")
        print(f"{'model':<{width}}  " + "  ".join(f"{c:>8}" for c in cols))
        for model, m in eval_results.items():
            print(
                f"{model:<{width}}  {m['val_nll']:8.4f}  {m['test_nll']:8.4f}"
                f"  {m['train_wall_time_s']:8.1f}  {m['inference_names_per_sec']:8.0f}"
            )


if __name__ == "__main__":
    main()
