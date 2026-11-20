"""Local name-generation UI (Streamlit app — basicConfig lives here, in the app).

    uv run streamlit run scripts/app_streamlit.py

Loads a bundle exported locally by `scripts/publish_hf.py --mode export`
(data/hf/models/<model>) or directly from a HF model repo ("hf:<repo_id>").
"""

import logging
from pathlib import Path

from onomatron.serialize import load_bundle, load_bundle_from_hub

try:
    import streamlit as st
except ImportError as e:
    raise SystemExit(
        "streamlit is a dev dependency — run `uv sync` first, then "
        "`uv run streamlit run scripts/app_streamlit.py`"
    ) from e

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path("data/hf/models")
HF_OPTION = "hf:<repo_id>…"


@st.cache_resource
def get_model(source: str):
    """source: a local bundle dir name under data/hf/models, or 'hf:<repo_id>'."""
    if source.startswith("hf:"):
        repo_id = source[3:]
        logger.info("loading model from HF repo %s", repo_id)
        return load_bundle_from_hub(repo_id)
    logger.info("loading local bundle %s", MODELS_DIR / source)
    return load_bundle(MODELS_DIR / source)


def main() -> None:
    st.set_page_config(page_title="onomatron", page_icon="📖")
    st.title("onomatron — character name generator")
    st.caption(
        "Character-level models trained on French literature character names "
        "(see the published HF repos)."
    )

    local = sorted(p.name for p in MODELS_DIR.iterdir()) if MODELS_DIR.is_dir() else []
    choice = st.selectbox(
        "Model",
        local + [HF_OPTION],
        help=f"local bundles in {MODELS_DIR}, or a HF repo",
    )
    if choice == HF_OPTION:
        choice = "hf:" + st.text_input(
            "HF repo", value="", placeholder="namespace/onomatron-transformer"
        )
    if not choice:
        st.info(
            "Pick a model (export first with `scripts/publish_hf.py --mode export`)."
        )
        return

    n = st.slider("Number of names", 1, 200, 20)
    seed = st.number_input("Seed", value=42, step=1)
    min_len = st.slider("Min length", 1, 10, 3)
    max_len = st.slider("Max length", 5, 40, 40)

    if st.button("Generate", type="primary"):
        with st.spinner("Generating…"):
            net, config = get_model(choice)
            names = net.sample(n=n, min_len=min_len, seed=int(seed), max_len=max_len)
        st.subheader(f"{len(names)} names (seed={seed})")
        st.code("\n".join(names), language=None)
        st.caption(f"model={config['model']} · params={config['params']}")


if __name__ == "__main__":
    main()
