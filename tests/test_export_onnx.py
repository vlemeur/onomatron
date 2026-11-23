"""Tests for scripts/export_onnx.py — skipped unless the 'export' extra is
installed (CI runs `uv sync --locked` without extras, so `onnx` is absent and
the whole module skips via importorskip)."""

import importlib.util
import json
from pathlib import Path

import pytest

from onomatron.data import SENTINEL, Vocab
from onomatron.models.mlp import MLP
from onomatron.models.rnn import RNNName

onnx = pytest.importorskip("onnx")

# scripts/ is not a package: load export_onnx.py by path.
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_onnx.py"
spec = importlib.util.spec_from_file_location("export_onnx", SCRIPT)
assert spec is not None and spec.loader is not None
export_onnx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export_onnx)

NAMES = ["cosette", "javert", "valjean", "marius", "éponine"]


@pytest.fixture
def vocab() -> Vocab:
    return Vocab(NAMES)


def _config(model: str, vocab: Vocab) -> dict:
    return {
        "model": model,
        "vocab": {"sentinel": SENTINEL, "stoi": vocab.stoi},
        "metrics": {"test_nll": 2.5},
    }


def test_export_rnn_step(tmp_path, vocab) -> None:
    net = RNNName(vocab, emb_dim=4, h_dim=8, seed=42)
    net.eval()
    onnx_path = export_onnx.export_onnx(net, "rnn", tmp_path)
    assert onnx_path == tmp_path / "model.onnx"
    assert onnx_path.stat().st_size < export_onnx.MAX_ONNX_BYTES
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    assert [i.name for i in model.graph.input] == ["tok", "h"]
    assert [o.name for o in model.graph.output] == ["logits", "h_new"]
    # the graph must be loop-free: only basic ops, safe for onnxruntime-web wasm
    op_types = {n.op_type for n in model.graph.node}
    assert "Loop" not in op_types and "Scan" not in op_types


def test_export_mlp_context(tmp_path, vocab) -> None:
    net = MLP(vocab, block_size=4, emb_dim=4, h_dim=8, seed=42)
    net.eval()
    onnx_path = export_onnx.export_onnx(net, "mlp", tmp_path)
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    assert [i.name for i in model.graph.input] == ["context"]
    assert [o.name for o in model.graph.output] == ["logits"]


def test_max_bytes_budget_enforced(tmp_path, vocab) -> None:
    net = RNNName(vocab, emb_dim=4, h_dim=8, seed=42)
    with pytest.raises(RuntimeError, match="budget"):
        export_onnx.export_onnx(net, "rnn", tmp_path, max_bytes=10)
    # the oversized artefact is removed, not left behind
    assert not (tmp_path / "model.onnx").exists()


def test_graph_kind_rejects_unsupported() -> None:
    with pytest.raises(ValueError, match="no ONNX export"):
        export_onnx.graph_kind("transformer")
    assert export_onnx.graph_kind("rnn") == "rnn-step"
    assert export_onnx.graph_kind("wavenet") == "ff-context"


def test_demo_vocab_json(vocab) -> None:
    net = RNNName(vocab, emb_dim=4, h_dim=8, seed=42)
    doc = export_onnx.demo_vocab_json(net, _config("rnn", vocab), "rnn-step")
    assert doc["format"] == export_onnx.DEMO_FORMAT
    assert doc["kind"] == "rnn-step"
    assert doc["sentinel"] == SENTINEL
    assert doc["n_vocab"] == len(vocab)
    assert doc["chars"][doc["stoi"][SENTINEL]] == SENTINEL
    assert doc["emb_dim"] == 4 and doc["h_dim"] == 8
    assert "block_size" not in doc  # rnn has no context window
    json.dumps(doc, ensure_ascii=False)  # must stay JSON-serializable
