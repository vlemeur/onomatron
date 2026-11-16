"""Hugging Face dataset descriptors. This is DATA, not code.

SourceSpec(kind="hf_dataset", target=<key>) selects an entry from DATASETS.
The sept-romans corpus is a hand-annotated gold standard (ObTIC, Sorbonne
University — Maurel et al., "Du repérage à l'analyse", DH 2025): NER
surface forms (aliases) resolved to canonical character entities, for
PER rows only. Rows: form (surface alias), type (PER/LOC/ORG), mentions,
entity (canonical character), novel.
"""

DATASETS: dict[str, dict[str, str]] = {
    "seven_novels_alias_resolution": {
        "repo_id": "compnet-renard/7-romans-alias-resolution",
        "config": "alias-resolution",
        "split": "train",
    },
}

# dataset "novel" identifier -> author, for per-row attribution.
# Keys are the exact `novel` values of the dataset.
NOVEL_AUTHORS: dict[str, str] = {
    "BelAmi": "Guy de Maupassant",
    "EugénieGrandet": "Honoré de Balzac",
    "Germinal": "Émile Zola",
    "LeRougeEtLeNoir": "Stendhal",
    "LesTroisMousquetaires": "Alexandre Dumas",
    "MadameBovary": "Gustave Flaubert",
    "NotreDameDeParis": "Victor Hugo",
}
