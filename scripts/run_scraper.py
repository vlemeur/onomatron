"""Scrape character names from French literature via the Wikipedia API.

Production orchestrator: iterates over every SourceSpec, extracts the
candidate names through the SAME dispatch as the probe, writes one JSONL
file per source (output_path()) and the consolidated dataset
data/interim/names.jsonl, deduplicated by normalized canonical (first
occurrence keeps its source — Wikipédia sources come first, so they win
over Wikidata). Ends with summary stats. A broken source is a line in
the report, not an error — same contract as the probe.
"""

import logging
from pathlib import Path

from onomatron.scraper import extract, sources, sparql_queries
from onomatron.scraper.fetch import SparqlClient, WikipediaClient
from onomatron.scraper.models import CharacterName, SourceSpec

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

INTERIM_DIR = Path("data/interim")
CONSOLIDATED_PATH = INTERIM_DIR / "names.jsonl"


def scrape_source(
    client: WikipediaClient, spec: SourceSpec, sparql_client: SparqlClient
) -> tuple[list[CharacterName], str]:
    """Extract names for one source. Never raises: a broken source is
    information, not an error — report it and move on."""
    try:
        # category members are listed via their own API: never fetch wikitext
        # for them (Catégorie:X targets have no article page).
        if spec.kind == "category":
            pages, _subcats = client.get_category_members(spec.target)
            names = []
            seen: set[str] = set()
            for page in pages:
                name = extract.title_to_name(
                    page, source=spec.target, author=spec.author
                )
                if name is None or name.canonical in seen:
                    continue
                seen.add(name.canonical)
                names.append(name)
            return names, "ok"

        if spec.kind == "sparql":
            try:
                query = sparql_queries.QUERIES[spec.target]
            except KeyError:
                return [], f"requête sparql inconnue: {spec.target}"
            rows = sparql_client.run(query)
            return extract.extract_from_sparql(rows, author=spec.author), "ok"

        wikitext = client.get_wikitext(spec.target)
        if spec.kind == "list_page":
            return extract.extract_from_list_page(wikitext, source=spec.target), "ok"

        if spec.kind == "work_page":
            if spec.section is None:
                return [], "work_page sans section"
            return (
                extract.extract_from_work_section(
                    wikitext,
                    source=spec.target,
                    author=spec.author,
                    section_heading=spec.section,
                ),
                "ok",
            )
    except ValueError as exc:
        return [], f"page introuvable ({exc})"
    except Exception as exc:  # network failure after retries
        return [], f"échec réseau ({type(exc).__name__})"
    return [], f"kind inconnu: {spec.kind}"


def write_names_jsonl(path: Path, names: list[CharacterName]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for name in names:
            fh.write(name.model_dump_json() + "\n")


def main() -> None:
    client = WikipediaClient()
    sparql_client = SparqlClient()

    # (spec, names, status) in SOURCES order — first occurrence wins on
    # consolidation, so source order in sources.py matters.
    results: list[tuple[SourceSpec, list[CharacterName], str]] = []
    for spec in sources.SOURCES:
        names, status = scrape_source(client, spec, sparql_client)
        write_names_jsonl(sources.output_path(spec), names)
        results.append((spec, names, status))
        logger.info("%s: %d names (%s)", spec.name, len(names), status)

    # ── Consolidation: dedupe by normalized canonical (lowercase, sans
    # ponctuation), first occurrence keeps its source — since Wikipédia
    # sources come first in SOURCES, they have priority over Wikidata.
    seen: set[str] = set()
    consolidated: list[CharacterName] = []
    duplicates = 0
    for _spec, names, _status in results:
        for name in names:
            key = extract.normalize_canonical(name.canonical)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            consolidated.append(name)

    write_names_jsonl(CONSOLIDATED_PATH, consolidated)

    # ── Stats résumées
    print(f"\nDataset consolidé: {CONSOLIDATED_PATH}")
    print(f"Total: {len(consolidated)} noms uniques ({duplicates} doublons éliminés)\n")

    per_author: dict[str | None, int] = {}
    for spec, names, _status in results:
        per_author[spec.author] = per_author.get(spec.author, 0) + len(names)
    width = max(len(a or "—") for a in per_author) + 2
    print(f"{'author':<{width}}{'names':>7} (par source, avant dédoublonnage)")
    for author, count in sorted(
        per_author.items(), key=lambda kv: -kv[1]
    ):  # sort by count desc
        print(f"{(author or '—'):<{width}}{count:>7}")

    failed = [(s, status) for s, _n, status in results if status != "ok"]
    if failed:
        print("\nSources en échec:")
        for spec, status in failed:
            print(f"  {spec.name}: {status}")


if __name__ == "__main__":
    main()
