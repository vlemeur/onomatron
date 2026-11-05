"""Configuration of scrape targets. This is DATA, not code."""

from pathlib import Path

from onomatron.scraper.models import SourceSpec

SOURCES: list[SourceSpec] = [
    SourceSpec(
        name="comedie_humaine",
        kind="list_page",
        target="Liste des personnages de La Comédie humaine",
        author="Honoré de Balzac",
    ),
    SourceSpec(
        name="personnages_fiction_francaise",
        kind="category",
        target="Personnage de fiction français",
    ),
    # Next milestones: Misérables, Proust, Germinal... one line each.
]


def output_path(spec: SourceSpec, root: str | Path = "data/interim") -> Path:
    return Path(root) / f"{spec.name}.jsonl"
