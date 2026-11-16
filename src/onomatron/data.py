"""Dataset loading and character vocabulary for the name models."""

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# Single begin/end-of-name character, index 0 in the vocab. Safe because the
# pipeline guarantees names.txt contains only latin letters and spaces.
SENTINEL = "."

DEFAULT_NAMES_PATH = Path("data/processed/names.txt")


def load_names(path: str | Path = DEFAULT_NAMES_PATH) -> list[str]:
    """Load the final names.txt (one name per line, already sanitized)."""
    names = [
        line
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    logger.info("loaded %d names from %s", len(names), path)
    return names


class Vocab:
    """Character <-> index mapping. The '.' sentinel is index 0 and marks
    both the beginning and the end of a name."""

    def __init__(self, names: list[str]) -> None:
        chars = sorted(set("".join(names)))
        self.stoi: dict[str, int] = {c: i + 1 for i, c in enumerate(chars)}
        self.stoi[SENTINEL] = 0
        self.itos: dict[int, str] = {i: s for s, i in self.stoi.items()}

    def __len__(self) -> int:
        return len(self.stoi)


def make_ngram_pairs(names: list[str], vocab: Vocab, n: int) -> torch.Tensor:
    """All n-grams of the names as index tuples: n-1 context chars + target.
    Each name is padded with n-1 sentinels at the start and one at the end.
    Returns a (N, n) int64 tensor; the last column is the target."""
    if n < 2:
        raise ValueError("n must be >= 2")
    pairs: list[tuple[int, ...]] = []
    for name in names:
        chs = [SENTINEL] * (n - 1) + list(name) + [SENTINEL]
        windows = zip(*(chs[i:] for i in range(n)), strict=False)
        pairs.extend(tuple(vocab.stoi[c] for c in w) for w in windows)
    return torch.tensor(pairs, dtype=torch.int64)
