# onomatron web demo

Static, serverless demo: a character-level RNN generating names of characters
from French literature, running **entirely in the browser** with
[onnxruntime-web](https://onnxruntime.ai/docs/get-started/with-javascript.html)
loaded from a CDN. No framework, no build step, no `node_modules`.

- `index.html` / `style.css` / `app.js` — the front (vanilla JS)
- `model.onnx` — the exported model (~32 KB), committed as a lightweight
  artefact
- `vocab.json` — sentinel + char↔index mapping + model dims + metrics, written
  by the export script

Type a prefix, click **Next char**: the model completes the name one
character per click (sampling from `softmax(logits / temperature)`), until
the end-of-name sentinel is drawn.

## Regenerate the artefacts

After a new benchmark / publish (i.e. a new best model under
`data/hf/models/`):

```bash
uv run --extra export python scripts/export_onnx.py          # rnn -> demo/
# or another exportable model:
uv run --extra export python scripts/export_onnx.py --model wavenet --out demo/
```

The script refuses to write a `model.onnx` of 10 MiB or more. The export is
checked (`onnx.checker`, torch-vs-ONNX numeric check when `onnxruntime` is
available) and `demo/vocab.json` is rewritten alongside.

## Run locally

```bash
cd demo && python -m http.server 8000
# open http://localhost:8000
```

(any static file server works; opening `index.html` via `file://` will not,
because browsers block `fetch("model.onnx")` on `file://`)

## Deployment

`.github/workflows/pages.yml` publishes this directory on every push to
`main` that touches `demo/`, using the official **GitHub Actions** Pages
source (Settings → Pages → Source: **GitHub Actions** — no `gh-pages` branch
involved): the build job uploads `demo/` with `actions/upload-pages-artifact`,
the deploy job publishes it with `actions/deploy-pages`. The demo is then
served at `https://<owner>.github.io/onomatron/`.
