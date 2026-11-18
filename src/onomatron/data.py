"""Dataset loading and character vocabulary for the name models."""

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# Single begin/end-of-name character, index 0 in the vocab. Safe because the
# pipeline guarantees names.txt contains only latin letters (full names are
# split on the space in Phase 2, so the space is never in the alphabet).
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


def split_names(
    names: list[str],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Shuffle `names` with a seeded generator and split into (train, val, test).

    The split happens on names, never on n-grams: n-grams of a single name are
    correlated, so cutting at the pair level would leak between splits. Ratios
    are normalized, so (8, 1, 1) works as well as (0.8, 0.1, 0.1).
    """
    total = sum(ratios)
    if len(ratios) != 3 or min(ratios) < 0 or total <= 0:
        raise ValueError("ratios must be 3 non-negative values summing to > 0")
    frac = tuple(r / total for r in ratios)
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(names), generator=gen).tolist()
    n_train = int(frac[0] * len(names))
    n_val = int(frac[1] * len(names))
    train = [names[i] for i in perm[:n_train]]
    val = [names[i] for i in perm[n_train : n_train + n_val]]
    test = [names[i] for i in perm[n_train + n_val :]]
    logger.info(
        "split %d names into %d train / %d val / %d test (seed=%d)",
        len(names),
        len(train),
        len(val),
        len(test),
        seed,
    )
    return train, val, test


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
