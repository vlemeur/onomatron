# onomatron

[![CI](https://github.com/vlemeur/onomatron/actions/workflows/ci.yml/badge.svg)](https://github.com/vlemeur/onomatron/actions/workflows/ci.yml)

A character-level language model that generates names of characters from French
literature, from the classics to the modern era — from scratch, in the spirit of
Karpathy's [makemore](https://github.com/karpathy/makemore) and his
[Zero to Hero](https://github.com/karpathy/nn-zero-to-hero) series.

## Layout

- `notebooks/` — the makemore journey, one notebook per architecture
- `src/onomatron/scraper/` — builds `data/processed/names.txt` (Wikipedia, Wikidata
  SPARQL, Hugging Face datasets)
- `src/onomatron/models/` — 11 models sharing one interface
  (`loss` / `nll` / `probs` / `sample` / `n_parameters`): `ngram`, `linear`,
  `mlp`, `wavenet`, `dilated_wavenet`, `rnn`, `lstm`, `gru`, `transformer`,
  `rwkv`, `mamba`
- `src/onomatron/benchmark/` — Optuna (TPE + ASHA) tuning + MLflow tracking of
  every trial and of the final evaluation
- `src/onomatron/publish/` — export/push of the dataset, the top models and the
  benchmark results to the Hugging Face Hub (`scripts/publish_hf.py`)
- `demo/` — serverless web demo on the ONNX export of the best model, published
  to GitHub Pages

## Usage

```bash
uv sync --locked
just scrape          # build the dataset (network)
just bench           # tune + evaluate all models
just bench-ui        # MLflow UI on http://127.0.0.1:5000
just run             # local UI on the best published model (fetched from the Hub)
just export-onnx     # export the best model to demo/model.onnx (ONNX, export extra)
just demo            # serve demo/ locally on http://localhost:8000
```

## Web demo

Serverless browser demo in `demo/` (vanilla JS + onnxruntime-web CDN, model
exported to ONNX): <https://vlemeur.github.io/onomatron/>. Regenerate the
artefacts with `just export-onnx`; deployed by `.github/workflows/pages.yml`.

Names are split **by name** (seed 42) into train/val/test = 80/10/10; the test
split is touched exactly once, in the final evaluation. Everything is tracked in
MLflow under `data/benchmark/`.

A local generation UI (Streamlit) is in `scripts/app_streamlit.py`.

## Results

Final evaluation on the held-out **test** split (touched exactly once), after
Optuna tuning (TPE + ASHA pruning) of each architecture. Mean ± std over 2
seeds (1 seed for rwkv and mamba); hardware: Apple Silicon (MPS). Full runs,
hyperparameters and curves are tracked in MLflow under `data/benchmark/`
(`just bench-ui`).

| model | test NLL | params | size | train time | inference |
|---|---:|---:|---:|---:|---:|
| rwkv | **2.345** | 121k | 0.48 MB | 37 s | 1 085 names/s |
| rnn | 2.349 | 7.7k | 0.03 MB | 53 s | 4 962 names/s |
| wavenet | 2.356 ± 0.010 | 51k | 0.20 MB | 16 s | 1 451 names/s |
| mamba | 2.365 | 56k | 0.22 MB | 717 s | 87 names/s |
| lstm | 2.368 ± 0.004 | 10k | 0.04 MB | 7 s | 3 791 names/s |
| mlp | 2.368 | 39k | 0.16 MB | 12 s | 7 711 names/s |
| transformer | 2.376 | 805k | 3.22 MB | 65 s | 219 names/s |
| gru | 2.381 ± 0.028 | 21k | 0.08 MB | 23 s | 3 667 names/s |
| dilated_wavenet | 2.433 ± 0.008 | 13k | 0.05 MB | 35 s | 441 names/s |
| ngram | 2.463 ± 0.000 | — (count table) | 0.44 MB | < 1 s | 19 094 names/s |
| linear | 2.593 | 1 444 | 0.01 MB | 4 s | 7 367 names/s |

RWKV edges out the vanilla RNN on test NLL (2.345 vs 2.349) but at 16× the
parameters and ~4.6× slower generation — for a name generator, the RNN is the
practical sweet spot. Every neural model beats the count-based baselines
(ngram 2.463, linear 2.593) by 0.1–0.25 nats. Mamba is the outlier:
competitive test NLL (2.365) but 12 minutes of training and 87 names/s. The
transformer is mid-pack despite 805k parameters (10× the next model) and the
second-slowest inference.

## Dataset

`data/processed/names.txt` — one sanitized name per line (~5.9k names). The ObTIC
subset declares no license: please cite Maurel et al., « Du repérage à l'analyse »,
DH 2025 (ObTIC, Sorbonne Université).

## License

Code and model weights: [MIT](LICENSE). The dataset carries per-source licenses
(Wikipedia CC BY-SA 4.0, Wikidata CC0, ObTIC without declared license) — see the
provenance card of the published `onomatron-names` dataset repo on the Hub
(`vlemeur/onomatron-names`; models under `vlemeur/onomatron-<model>`).

## Citation

```bibtex
@misc{onomatron2026,
  author = {Le Meur, Vincent},
  title  = {onomatron: character-level generation of French literature names},
  year   = {2026},
  url    = {https://github.com/vlemeur/onomatron},
}
```
