"""Tests for onomatron.data: vocab mapping, name splitting, tensor builders."""

from pathlib import Path
from typing import cast

import pytest
import torch

from onomatron.data import (
    SENTINEL,
    Vocab,
    load_names,
    make_ngram_pairs,
    make_sequences,
    split_names,
)


class TestVocab:
    def test_sentinel_is_index_zero(self, vocab: Vocab) -> None:
        assert vocab.stoi[SENTINEL] == 0
        assert vocab.itos[0] == SENTINEL

    def test_stoi_and_itos_are_inverse(self, vocab: Vocab) -> None:
        for ch, ix in vocab.stoi.items():
            assert vocab.itos[ix] == ch

    def test_len_counts_the_sentinel(self, names: list[str]) -> None:
        assert len(Vocab(names)) == len(set("".join(names))) + 1

    def test_from_stoi_roundtrip(self, vocab: Vocab) -> None:
        rebuilt = Vocab.from_stoi(vocab.stoi)
        assert rebuilt.stoi == vocab.stoi
        assert rebuilt.itos == vocab.itos


class TestLoadNames:
    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "names.txt"
        path.write_text("cosette\n\njavert\n   \n", encoding="utf-8")
        assert load_names(path) == ["cosette", "javert"]


class TestSplitNames:
    def test_same_seed_same_split(self, names: list[str]) -> None:
        assert split_names(names, seed=42) == split_names(names, seed=42)

    def test_different_seed_shuffles(self, names: list[str]) -> None:
        assert split_names(names, seed=0) != split_names(names, seed=1)

    def test_splits_are_disjoint_and_complete(self, names: list[str]) -> None:
        train, val, test = split_names(names, seed=0)
        assert sorted(train + val + test) == sorted(names)
        assert set(train).isdisjoint(val)
        assert set(train).isdisjoint(test)
        assert set(val).isdisjoint(test)

    def test_ratios_are_normalized(self, names: list[str]) -> None:
        assert split_names(names, ratios=(0.8, 0.1, 0.1), seed=7) == split_names(
            names, ratios=(8, 1, 1), seed=7
        )

    def test_default_split_sizes(self, names: list[str]) -> None:
        train, val, test = split_names(names, seed=0)
        assert len(train) == int(0.8 * len(names))
        assert len(val) == int(0.1 * len(names))
        assert len(test) == len(names) - len(train) - len(val)

    @pytest.mark.parametrize(
        "ratios",
        [
            (0.5, 0.5),  # not 3 values
            (0.8, 0.1, 0.1, 0.0),  # 4 values
            (-0.1, 0.6, 0.5),  # negative
            (0.0, 0.0, 0.0),  # zero sum
        ],
    )
    def test_invalid_ratios_raise(
        self, names: list[str], ratios: tuple[float, ...]
    ) -> None:
        with pytest.raises(ValueError, match="ratios"):
            # the invalid cases are deliberately not 3-tuples of valid floats
            bad = cast("tuple[float, float, float]", ratios)
            split_names(names, ratios=bad)


class TestMakeNgramPairs:
    def test_bigram_windows_of_a_two_letter_name(self) -> None:
        vocab = Vocab(["ab"])  # sorted chars: a=1, b=2, sentinel=0
        pairs = make_ngram_pairs(["ab"], vocab, 2)
        assert pairs.dtype == torch.int64
        assert pairs.tolist() == [[0, 1], [1, 2], [2, 0]]

    def test_shape_is_ngrams_by_n(self, names: list[str], vocab: Vocab) -> None:
        n = 4
        pairs = make_ngram_pairs(names, vocab, n)
        assert pairs.shape[1] == n
        # one window per position of each sentinel-padded name
        assert pairs.shape[0] == sum(len(name) + 1 for name in names)

    def test_n_below_two_raises(self, names: list[str], vocab: Vocab) -> None:
        with pytest.raises(ValueError, match="n must be"):
            make_ngram_pairs(names, vocab, 1)


class TestMakeSequences:
    def test_x_predicts_y_position_by_position(self) -> None:
        vocab = Vocab(["ab"])
        x, y = make_sequences(["ab"], vocab)
        # padded seq = [., a, b, .] -> x = [., a, b] (+ sentinel pad),
        # y = [a, b, .] (+ -100 pad)
        assert x[0, :3].tolist() == [0, 1, 2]
        assert y[0, :3].tolist() == [1, 2, 0]

    def test_padding_marks(self, names: list[str], vocab: Vocab) -> None:
        x, y = make_sequences(names, vocab)
        for i, name in enumerate(names):
            n_real = len(name) + 1  # inputs = seq[:-1]
            # y is padded with -100 (ignored by cross_entropy)...
            assert bool((y[i, n_real:] == -100).all())
            # ...while x is padded with the sentinel index (a valid row)
            assert bool((x[i, n_real:] == vocab.stoi[SENTINEL]).all())
            # the real end-of-name target (sentinel) is not masked
            assert y[i, n_real - 1].item() == vocab.stoi[SENTINEL]
