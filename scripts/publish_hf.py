"""Publish onomatron artifacts to the Hugging Face Hub (thin CLI).

Export only (local, no network) — assembles data/hf/ for inspection:

    uv run python scripts/publish_hf.py --mode export --top-n 3

Push (needs HF_TOKEN with write access + a namespace):

    HF_TOKEN=hf_... uv run python scripts/publish_hf.py --mode push \
        --namespace <you> --top-n 3

Requires the benchmark to have run at least in eval mode (MLflow runs under
data/benchmark/). Repos created: <namespace>/onomatron-names (dataset),
onomatron-benchmark (results + tracking DBs), onomatron-<model> per top model.
"""

import argparse
import logging
import sys

from onomatron.publish import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["export", "push"], default="export")
    parser.add_argument(
        "--namespace", default=None, help="HF user or org (required for --mode push)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="number of best models to publish, ranked by val NLL (default 3)",
    )
    parser.add_argument(
        "--models", nargs="+", default=None, help="restrict to these models"
    )
    parser.add_argument("--out-root", default="data/hf", help="local export directory")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )
    if args.mode == "push" and not args.namespace:
        parser.error("--namespace is required with --mode push")

    summary = run(
        namespace=args.namespace,
        top_n=args.top_n,
        mode=args.mode,
        out_root=args.out_root,
        models=args.models,
    )
    print(
        f"\n{summary['n_runs']} eval runs found; top {len(summary['top'])} by val NLL:"
    )
    print("  " + ", ".join(summary["top"]))
    print(f"exported under {summary['out_root']}/ (manifest.json has the details)")
    if summary["pushed"]:
        print("pushed to HF:")
        for repo in summary["pushed"]:
            print(f"  https://huggingface.co/{repo}")
    else:
        print("local export only — to push:")
        print(
            "  HF_TOKEN=hf_... uv run python scripts/publish_hf.py "
            "--mode push --namespace <you> --top-n 3"
        )


if __name__ == "__main__":
    main()
