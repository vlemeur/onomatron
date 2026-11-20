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

Génère des **noms de personnages de la littérature française** avec un modèle
de langage caractère par caractère entraîné from scratch (esprit makemore).

Ce répertoire dans le repo git ne contient que `app.py` + `requirements.txt`
+ ce README : au moment du push (`scripts/publish_hf.py --mode push`), le
package `onomatron/` (sous-ensemble torch-only vendored) y est ajouté, et la
variable d'environnement `ONOMATRON_MODEL_REPO` du Space est renseignée avec
le meilleur modèle publié.

En local, sans HF :

```bash
uv run streamlit run scripts/app_streamlit.py
```
