---
title: onomatron
emoji: 📖
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# onomatron

Generates **names of characters from French literature** with a from-scratch
character-level language model (makemore spirit).

This directory in the git repo only holds `app.py` + `requirements.txt` +
this README: at push time (`scripts/publish_hf.py --mode push`), the vendored
torch-only `onomatron/` package is added here, and the Space variable
`ONOMATRON_MODEL_REPO` is set to the best published model.

For a local interface instead of the Space:

```bash
uv run streamlit run scripts/app_streamlit.py
```
