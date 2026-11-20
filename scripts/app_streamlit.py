"""Local name-generation UI (Streamlit) on the best model published on the Hub.

    just run
    # or: uv run streamlit run scripts/app_streamlit.py

The default repo is the benchmark's best model (lowest val NLL) published under
the ONOMATRON_NAMESPACE namespace (default: Jig09); the weights are fetched from
the Hub at launch (then cached). Override the repo in the text field.
"""

import logging
import os

from onomatron.serialize import load_bundle_from_hub

try:
    import streamlit as st
except ImportError as e:
    raise SystemExit(
        "streamlit is a dev dependency — run `uv sync` first, then `just run`"
    ) from e

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

NAMESPACE = os.environ.get("ONOMATRON_NAMESPACE", "Jig09")
FALLBACK_MODEL = "rnn"


def default_repo() -> str:
    """Repo of the best published model: best val NLL of the local benchmark
    state, with a constant fallback when no benchmark has run locally."""
    try:
        from onomatron.benchmark.config import BenchmarkConfig
        from onomatron.publish.export import best_model_name

        model = best_model_name(BenchmarkConfig())
    except Exception as e:
        logger.warning(
            "no local benchmark state (%s) — fallback to %s", e, FALLBACK_MODEL
        )
        model = FALLBACK_MODEL
    return f"{NAMESPACE}/onomatron-{model}"


@st.cache_resource
def get_model(repo_id: str):
    logger.info("fetching model from HF repo %s", repo_id)
    return load_bundle_from_hub(repo_id)


def main() -> None:
    st.set_page_config(page_title="onomatron", page_icon="📖")
    st.title("onomatron — character name generator")
    st.caption(
        "Character-level model trained on French literature character names, "
        "fetched from the Hugging Face Hub."
    )
    repo = st.text_input("HF model repo", value=default_repo())
    n = st.slider("Number of names", 1, 200, 20)
    seed = st.number_input("Seed", value=42, step=1)
    min_len = st.slider("Min length", 1, 10, 3)
    max_len = st.slider("Max length", 5, 40, 40)

    if st.button("Generate", type="primary"):
        if not repo.strip():
            st.info("Enter a HF model repo, e.g. Jig09/onomatron-rnn")
            return
        with st.spinner("Fetching model + generating…"):
            net, config = get_model(repo.strip())
            names = net.sample(n=n, min_len=min_len, seed=int(seed), max_len=max_len)
        st.subheader(f"{len(names)} names (seed={seed})")
        st.code("\n".join(names), language=None)
        st.caption(
            f"model={config['model']} · repo={repo.strip()} · params={config['params']}"
        )


if __name__ == "__main__":
    main()
