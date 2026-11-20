"""Round-trip tests: a saved bundle must rebuild a model that behaves exactly
like the original — same probabilities, same seeded samples — so a published
bundle is truly self-contained."""

import json
from pathlib import Path

import pytest
import torch

from onomatron.data import Vocab, make_ngram_pairs
from onomatron.models.mlp import MLP
from onomatron.models.ngram import NGramModel
from onomatron.serialize import BUNDLE_FORMAT, load_bundle, save_bundle

NAMES = ["cosette", "javert", "marius"]


@pytest.fixture
def vocab() -> Vocab:
    return Vocab(NAMES)


def test_nn_module_roundtrip(tmp_path: Path, vocab: Vocab) -> None:
    model = MLP(vocab, block_size=3, emb_dim=4, h_dim=16, seed=0)
    pairs = make_ngram_pairs(NAMES, vocab, 4)
    x, _y = pairs[:, :-1], pairs[:, -1]
    save_bundle(
        tmp_path / "bundle",
        model="mlp",
        net=model,
        params={"block_size": 3, "emb_dim": 4, "h_dim": 16, "batch_norm": True},
        vocab=vocab,
        seed=0,
        metrics={"val_nll": 1.5},
    )

    net, config = load_bundle(tmp_path / "bundle")

    assert config["format"] == BUNDLE_FORMAT
    assert config["model"] == "mlp"
    assert config["class"] == "MLP"
    assert config["vocab"]["stoi"] == vocab.stoi
    assert config["metrics"] == {"val_nll": 1.5}
    assert not net.training  # eval mode after load
    assert torch.allclose(net.probs(x), model.probs(x), atol=1e-6)
    assert net.sample(n=5, seed=42) == model.sample(n=5, seed=42)


def test_ngram_roundtrip(tmp_path: Path, vocab: Vocab) -> None:
    model = NGramModel.fit(NAMES, vocab, n=3, smoothing=0.5)
    save_bundle(
        tmp_path / "bundle",
        model="ngram",
        net=model,
        params={"n": 3, "smoothing": 0.5},
        vocab=vocab,
        seed=None,
    )

    net, _config = load_bundle(tmp_path / "bundle")

    assert torch.equal(net.counts, model.counts)
    assert torch.allclose(net.probs, model.probs)
    assert net.sample(n=5, seed=1) == model.sample(n=5, seed=1)


def test_bundle_contains_exactly_two_files(tmp_path: Path, vocab: Vocab) -> None:
    model = NGramModel.fit(NAMES, vocab, n=2)
    out = save_bundle(
        tmp_path / "bundle",
        model="ngram",
        net=model,
        params={"n": 2, "smoothing": 1.0},
        vocab=vocab,
        seed=None,
    )
    assert sorted(p.name for p in out.iterdir()) == ["config.json", "model.pt"]
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["vocab"]["sentinel"] == "."


def test_unknown_bundle_format_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"format": "something-else/9"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unsupported bundle format"):
        load_bundle(tmp_path)
