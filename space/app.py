"""onomatron HF Space — generate French literature character names (Gradio).

The model repo is read from the ONOMATRON_MODEL_REPO Space variable, set by
scripts/publish_hf.py --mode push (default: the benchmark's best model).
The onomatron/ package next to this file is a vendored torch-only subset
added at publish time (see onomatron.publish.export.vendor_onomatron).
"""

import os
from functools import lru_cache

import gradio as gr

from onomatron.serialize import load_bundle_from_hub

DEFAULT_REPO = os.environ.get("ONOMATRON_MODEL_REPO", "")


@lru_cache(maxsize=4)
def get_model(repo_id: str):
    if not repo_id:
        raise RuntimeError(
            "Set the ONOMATRON_MODEL_REPO Space variable to a published model "
            "repo, e.g. '<namespace>/onomatron-transformer'."
        )
    return load_bundle_from_hub(repo_id)


def generate(repo_id: str, n: int, seed: float, min_len: int, max_len: int):
    net, config = get_model(repo_id.strip())
    names = net.sample(
        n=int(n), min_len=int(min_len), seed=int(seed), max_len=int(max_len)
    )
    meta = f"model `{config['model']}` · params `{config['params']}` · seed {int(seed)}"
    return "\n".join(names), meta


with gr.Blocks(title="onomatron") as demo:
    gr.Markdown(
        "# onomatron 📖\n"
        "Generates **names of characters from French literature** with a "
        "from-scratch character-level language model "
        "([makemore](https://github.com/karpathy/makemore) spirit)."
    )
    repo = gr.Textbox(
        label="Model repo",
        value=DEFAULT_REPO,
        placeholder="<namespace>/onomatron-<model>",
        info="Published by onomatron (ONOMATRON_MODEL_REPO Space variable).",
    )
    with gr.Row():
        n = gr.Slider(1, 200, value=20, step=1, label="Number of names")
        seed = gr.Number(value=42, label="Seed", precision=0)
    with gr.Row():
        min_len = gr.Slider(1, 10, value=3, step=1, label="Min length")
        max_len = gr.Slider(5, 40, value=40, step=1, label="Max length")
    btn = gr.Button("Generate", variant="primary")
    out = gr.Textbox(label="Generated names", lines=14)
    meta = gr.Markdown()
    btn.click(generate, inputs=[repo, n, seed, min_len, max_len], outputs=[out, meta])

demo.launch()
