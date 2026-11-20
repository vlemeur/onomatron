"""Generate names with a published onomatron model.

From this repo's root (any env with torch + huggingface_hub installed):

    python generate.py --bundle . --n 20 --seed 42

or directly from the Hub, without cloning:

    python generate.py --repo <namespace>/onomatron-<model> --n 20 --seed 42

This file lives in the onomatron repo under templates/ and is copied
verbatim into every published model repo by scripts/publish_hf.py.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from onomatron.serialize import load_bundle, load_bundle_from_hub


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=".", help="local bundle directory")
    parser.add_argument("--repo", default=None, help="HF repo id (instead of --bundle)")
    parser.add_argument("--n", type=int, default=20, help="number of names")
    parser.add_argument("--min-len", type=int, default=3)
    parser.add_argument("--max-len", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.repo:
        net, _config = load_bundle_from_hub(args.repo)
    else:
        net, _config = load_bundle(args.bundle)
    names = net.sample(
        n=args.n, min_len=args.min_len, seed=args.seed, max_len=args.max_len
    )
    print("\n".join(names))


if __name__ == "__main__":
    main()
