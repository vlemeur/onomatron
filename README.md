# onomatron

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

## Usage

```bash
uv sync --locked
just scrape          # build the dataset (network)
just bench           # tune + evaluate all models
just bench-ui        # MLflow UI on http://127.0.0.1:5000
```

Names are split **by name** (seed 42) into train/val/test = 80/10/10; the test
split is touched exactly once, in the final evaluation. Everything is tracked in
MLflow under `data/benchmark/`.

A local generation UI (Streamlit) is in `scripts/app_streamlit.py`.

## Dataset

`data/processed/names.txt` — one sanitized name per line (~5.9k names). The ObTIC
subset declares no license: please cite Maurel et al., « Du repérage à l'analyse »,
DH 2025 (ObTIC, Sorbonne Université).

## License

Code and model weights: [MIT](LICENSE). The dataset carries per-source licenses
(Wikipedia CC BY-SA 4.0, Wikidata CC0, ObTIC without declared license) — see the
provenance card of the published `onomatron-names` dataset repo on the Hub
(`Jig09/onomatron-names`; models under `Jig09/onomatron-<model>`).

## Citation

```bibtex
@misc{onomatron2026,
  author = {Le Meur, Vincent},
  title  = {onomatron: character-level generation of French literature names},
  year   = {2026},
  url    = {https://github.com/vlemeur/onomatron},
}
```
