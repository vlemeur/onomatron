"""Export a trained onomatron model to ONNX for the in-browser demo (demo/).

Local only — no network. Reads an existing model bundle (``data/hf/models/
<model>/``, written by ``scripts/publish_hf.py --mode export``) or rebuilds
the model from the MLflow eval run when the bundle is missing, then writes
into the output directory (default ``demo/``):

- ``model.onnx``: one batch-dynamic ONNX graph;
- ``vocab.json``: sentinel + stoi + model dims + metrics, for the JS front.

Two graph kinds:

- ``rnn-step`` (rnn): ``step(tok (N,1) int64, h (1,N,h_dim) f32) ->
  (logits (N,V), h_new (1,N,h_dim))`` — the RNN cell is unrolled manually, so
  the graph holds only Embedding/Gemm/Add/Tanh (no Loop op, safest for
  onnxruntime-web's wasm backend). The browser keeps ``h`` and calls one step
  per generated character;
- ``ff-context`` (mlp, wavenet, dilated_wavenet): ``(context (N,block_size)
  int64) -> logits (N,V)`` — one call per generated character on the rolling
  context window.

    uv run --extra export python scripts/export_onnx.py                  # rnn -> demo/
    uv run --extra export python scripts/export_onnx.py --model wavenet
    uv run --extra export python scripts/export_onnx.py --model auto     # best val NLL

Requires the ``export`` extra (``onnx``); the export fails if the resulting
``model.onnx`` is >= ``--max-bytes`` (default 10 MiB, the committed-artefact
budget).
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

from onomatron.data import Vocab, load_names
from onomatron.serialize import load_bundle

try:
    import onnx
except ImportError as e:
    raise SystemExit(
        "the ONNX exporter needs the 'export' extra — "
        "run: uv run --extra export python scripts/export_onnx.py"
    ) from e

logger = logging.getLogger(__name__)

DEMO_FORMAT = "onomatron-demo/1"
ONNX_MODEL_NAME = "model.onnx"
VOCAB_NAME = "vocab.json"
DEFAULT_OUT = Path("demo")
DEFAULT_BUNDLE_ROOT = Path("data/hf/models")
# Hard budget for the committed artefact (GitHub Pages payload).
MAX_ONNX_BYTES = 10 * 1024 * 1024
DEFAULT_OPSET = 18

# Models exportable to ONNX today, by graph kind.
STEP_MODELS = frozenset({"rnn"})
FF_MODELS = frozenset({"mlp", "wavenet", "dilated_wavenet"})


class RNNCellStep(nn.Module):
    """One step of the vanilla RNN, unrolled manually (no nn.RNN, hence no
    Loop op in the ONNX graph).

    tok: (N, 1) int64 char indices; h: (1, N, h_dim) float32 hidden state.
    Returns (logits (N, n_vocab), h_new (1, N, h_dim)).
    """

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.embedding = net.embedding
        self.output = net.output
        rnn = net.rnn
        self.register_buffer("w_ih", rnn.weight_ih_l0)
        self.register_buffer("w_hh", rnn.weight_hh_l0)
        self.register_buffer("b_ih", rnn.bias_ih_l0)
        self.register_buffer("b_hh", rnn.bias_hh_l0)

    def forward(
        self, tok: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(tok).squeeze(1)  # (N, emb_dim)
        hh = h.squeeze(0)  # (N, h_dim)
        pre = x @ self.w_ih.T + self.b_ih + hh @ self.w_hh.T + self.b_hh
        h_new = torch.tanh(pre)  # (N, h_dim)
        return self.output(h_new), h_new.unsqueeze(0)  # (N, V), (1, N, h_dim)


def graph_kind(model: str) -> str:
    """ONNX graph kind of a model: 'rnn-step' or 'ff-context'."""
    if model in STEP_MODELS:
        return "rnn-step"
    if model in FF_MODELS:
        return "ff-context"
    raise ValueError(
        f"model {model!r} has no ONNX export — supported: "
        f"{sorted(STEP_MODELS | FF_MODELS)}"
    )


def check_step_wrapper(net: nn.Module, cell: RNNCellStep) -> None:
    """Prove the unrolled cell is exactly the model's forward on one step."""
    torch.manual_seed(0)
    tok = torch.tensor([[0], [5], [12]], dtype=torch.int64)
    h = torch.randn(1, 3, net.h_dim)
    with torch.no_grad():
        logits_ref, h_ref = net(tok, h)
        logits_cell, h_cell = cell(tok, h)
    if not torch.allclose(logits_cell, logits_ref[:, -1], atol=1e-6):
        raise RuntimeError("unrolled RNN cell logits differ from net.forward")
    if not torch.allclose(h_cell, h_ref, atol=1e-6):
        raise RuntimeError("unrolled RNN cell hidden state differs from net.forward")
    logger.info("unrolled RNN cell matches net.forward (atol=1e-6)")


def load_net_from_mlflow(model: str) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild (net, config) of a model from its MLflow eval run, when the
    exported bundle under data/hf/models/ is missing."""
    from onomatron.benchmark.config import BenchmarkConfig
    from onomatron.benchmark.dataset import MODEL_CLASSES
    from onomatron.publish.export import arch_params, load_eval_runs
    from onomatron.serialize import make_config

    cfg = BenchmarkConfig()
    runs = load_eval_runs(cfg)
    run = next((r for r in runs if r.model == model), None)
    if run is None:
        raise RuntimeError(
            f"no eval run for model {model!r} in {cfg.mlflow_tracking_uri} — "
            "run `uv run python scripts/benchmark.py --mode eval` first"
        )
    weights = run.artifact_dir / "model.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"no model.pt in the artifacts of run {run.run_id}")
    vocab = Vocab(load_names(cfg.names_path))
    params = arch_params(run)
    seed = int(run.params.get("seed", cfg.seed))
    cls = MODEL_CLASSES[model]
    net = cls(vocab, seed=seed, **params)
    net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    net.eval()
    metrics = {k: run.metrics[k] for k in run.metrics if k.endswith("_nll")}
    config = make_config(model, cls.__name__, params, vocab, seed, metrics=metrics)
    logger.info("rebuilt %s from MLflow run %s", model, run.run_id)
    return net, config


def resolve_model(requested: str) -> str:
    """'auto' -> name of the best model of the local benchmark state."""
    if requested != "auto":
        return requested
    from onomatron.publish.export import best_model_name

    model = best_model_name()
    logger.info("auto model selection (best val NLL): %s", model)
    return model


def demo_vocab_json(
    net: nn.Module, config: dict[str, Any], kind: str
) -> dict[str, Any]:
    """vocab.json content consumed by demo/app.js."""
    stoi: dict[str, int] = config["vocab"]["stoi"]
    doc: dict[str, Any] = {
        "format": DEMO_FORMAT,
        "model": config["model"],
        "kind": kind,
        "sentinel": config["vocab"]["sentinel"],
        "stoi": stoi,
        # index -> char, as a plain list (itos)
        "chars": [c for c, _ in sorted(stoi.items(), key=lambda kv: kv[1])],
        "n_vocab": len(stoi),
    }
    for key in ("block_size", "emb_dim", "h_dim"):
        value = getattr(net, key, None)
        if value is not None:
            doc[key] = value
    if config.get("metrics"):
        doc["metrics"] = config["metrics"]
    return doc


def export_onnx(
    net: nn.Module,
    model: str,
    out_dir: str | Path,
    *,
    opset: int = DEFAULT_OPSET,
    max_bytes: int = MAX_ONNX_BYTES,
) -> Path:
    """Export `net` to <out_dir>/model.onnx (batch-dynamic) + vocab.json, with
    structural, size and (when onnxruntime is available) numeric checks."""
    kind = graph_kind(model)
    net.eval()
    if kind == "rnn-step":
        wrapper: nn.Module = RNNCellStep(net)
        check_step_wrapper(net, wrapper)  # type: ignore[arg-type]
        args = (
            torch.zeros((1, 1), dtype=torch.int64),
            torch.zeros((1, 1, net.h_dim), dtype=torch.float32),
        )
        input_names = ["tok", "h"]
        output_names = ["logits", "h_new"]
        dynamic_axes = {
            "tok": {0: "batch"},
            "h": {1: "batch"},
            "logits": {0: "batch"},
            "h_new": {1: "batch"},
        }
    else:  # ff-context: the model's forward is already (N, block_size) -> logits
        wrapper = net
        args = (torch.zeros((1, net.block_size), dtype=torch.int64),)
        input_names = ["context"]
        output_names = ["logits"]
        dynamic_axes = {"context": {0: "batch"}, "logits": {0: "batch"}}

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    onnx_path = out / ONNX_MODEL_NAME
    torch.onnx.export(
        wrapper,
        args,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        dynamo=False,
    )
    logger.info("exported %s (%s, opset %d)", onnx_path, kind, opset)

    model_proto = onnx.load(str(onnx_path))
    onnx.checker.check_model(model_proto)
    size = onnx_path.stat().st_size
    if size >= max_bytes:
        onnx_path.unlink()
        raise RuntimeError(
            f"model.onnx is {size / 1e6:.2f} MB >= {max_bytes / 1e6:.0f} MB budget — "
            "reduce the architecture or re-export a smaller model"
        )
    logger.info(
        "onnx.checker OK, size %.1f KB (< %.0f MB)", size / 1e3, max_bytes / 1e6
    )

    numeric_check_onnx(onnx_path, wrapper, args, input_names, output_names)
    return onnx_path


def numeric_check_onnx(
    onnx_path: Path,
    wrapper: nn.Module,
    args: tuple[torch.Tensor, ...],
    input_names: list[str],
    output_names: list[str],
) -> None:
    """Run the exported graph with onnxruntime (when installed) and compare
    against the torch wrapper on the dummy inputs."""
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        logger.warning(
            "onnxruntime not installed — skipping the numeric ONNX check "
            "(run with: uv run --extra export --with onnxruntime ...)"
        )
        return
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feeds = {name: t.numpy() for name, t in zip(input_names, args, strict=True)}
    got = sess.run(output_names, feeds)
    with torch.no_grad():
        ref = wrapper(*args)
    for name, r, g in zip(output_names, ref, got, strict=True):
        if not np.allclose(r.numpy(), g, atol=1e-4):
            raise RuntimeError(f"ONNX output {name!r} differs from torch")
    logger.info("numeric check OK: ONNX outputs match torch (atol=1e-4)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="rnn",
        help="model to export: rnn (default), wavenet, mlp, dilated_wavenet, "
        "or 'auto' for the best val NLL of the local benchmark",
    )
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=DEFAULT_BUNDLE_ROOT,
        help="root of the exported bundles (default: data/hf/models)",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="output directory (default: demo)"
    )
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=MAX_ONNX_BYTES,
        help="fail if model.onnx reaches this size (default: 10 MiB)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )

    model = resolve_model(args.model)
    bundle_dir = args.bundle_root / model
    if bundle_dir.is_dir():
        net, config = load_bundle(bundle_dir)
        logger.info("loaded bundle %s", bundle_dir)
    else:
        logger.warning(
            "no bundle at %s — rebuilding %s from the MLflow eval run",
            bundle_dir,
            model,
        )
        net, config = load_net_from_mlflow(model)

    onnx_path = export_onnx(
        net, model, args.out, opset=args.opset, max_bytes=args.max_bytes
    )
    kind = graph_kind(model)
    vocab_doc = demo_vocab_json(net, config, kind)
    vocab_path = args.out / VOCAB_NAME
    vocab_path.write_text(
        json.dumps(vocab_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    size = onnx_path.stat().st_size
    print(
        f"demo artefacts written to {args.out}/:\n"
        f"  {ONNX_MODEL_NAME}  {size / 1e3:.1f} KB (model={model}, kind={kind}, "
        f"opset={args.opset})\n"
        f"  {VOCAB_NAME}  vocab={vocab_doc['n_vocab']} chars"
    )


if __name__ == "__main__":
    main()
