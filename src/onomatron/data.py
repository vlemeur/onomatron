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


def make_bigram_pairs(names: list[str], vocab: Vocab) -> torch.Tensor:
    """All (context, next) index pairs of the names, each name padded with
    the sentinel on both sides. Returns a (N, 2) int64 tensor."""
    pairs: list[tuple[int, int]] = []
    for name in names:
        chs = [SENTINEL, *name, SENTINEL]
        pairs.extend(
            (vocab.stoi[a], vocab.stoi[b]) for a, b in zip(chs, chs[1:], strict=False)
        )
    return torch.tensor(pairs, dtype=torch.int64)
