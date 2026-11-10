"""Scrape character names from French literature via the Wikipedia API.

Production orchestrator: iterates over every SourceSpec, extracts the
candidate names through the SAME dispatch as the probe, writes one JSONL
file per source (output_path()) and the consolidated dataset
data/interim/names.jsonl, deduplicated by normalized canonical (first
occurrence keeps its source — Wikipedia sources come first, so they win
over Wikidata). Ends with summary stats. A broken source is a line in
the report, not an error — same contract as the probe.
"""

import logging
from pathlib import Path

from tqdm import tqdm

from onomatron.scraper import extract, hf_datasets, sources, sparql_queries
from onomatron.scraper.fetch import HuggingFaceClient, SparqlClient, WikipediaClient
from onomatron.scraper.models import CharacterName, SourceSpec

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

INTERIM_DIR = Path("data/interim")
CONSOLIDATED_PATH = INTERIM_DIR / "names.jsonl"


def scrape_source(
    client: WikipediaClient,
    spec: SourceSpec,
    sparql_client: SparqlClient,
    hf_client: HuggingFaceClient,
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
                return [], f"unknown sparql query: {spec.target}"
            rows = sparql_client.run(query)
            return extract.extract_from_sparql(rows, author=spec.author), "ok"

        if spec.kind == "hf_dataset":
            try:
                meta = hf_datasets.DATASETS[spec.target]
            except KeyError:
                return [], f"unknown hf dataset: {spec.target}"
            rows = hf_client.load_table(
                meta["repo_id"], meta["config"], meta.get("split", "train")
            )
            return (
                extract.extract_from_ner_rows(
                    rows, novel_authors=hf_datasets.NOVEL_AUTHORS
                ),
                "ok",
            )

        wikitext = client.get_wikitext(spec.target)
        if spec.kind == "list_page":
            return extract.extract_from_list_page(wikitext, source=spec.target), "ok"

        if spec.kind == "work_page":
            if spec.section is None:
                return [], "work_page without a section"
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
        return [], f"page not found ({exc})"
    except Exception as exc:  # network failure after retries
        return [], f"network failure ({type(exc).__name__})"
    return [], f"unknown kind: {spec.kind}"


def write_names_jsonl(path: Path, names: list[CharacterName]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for name in names:
            fh.write(name.model_dump_json() + "\n")


def main() -> None:
    client = WikipediaClient()
    sparql_client = SparqlClient()
    hf_client = HuggingFaceClient()

    # (spec, names, status) in SOURCES order — first occurrence wins on
    # consolidation, so source order in sources.py matters. The progress
    # bar shows the source being scraped and the running name count;
    # failures are written through the bar without breaking it.
    results: list[tuple[SourceSpec, list[CharacterName], str]] = []
    total = 0
    with tqdm(sources.SOURCES, desc="scrape", unit="source") as bar:
        for spec in bar:
            bar.set_postfix_str(f"{spec.name} · {total} names")
            names, status = scrape_source(client, spec, sparql_client, hf_client)
            write_names_jsonl(sources.output_path(spec), names)
            results.append((spec, names, status))
            total += len(names)
            if status != "ok":
                bar.write(f"failed {spec.name}: {status}")

    # ── Consolidation: dedupe by normalized canonical (lowercase,
    # punctuation stripped), first occurrence keeps its source — since
    # Wikipedia sources come first in SOURCES, they have priority over
    # Wikidata.
    seen: set[str] = set()
    consolidated: list[CharacterName] = []
    duplicates = 0
    for _, names, _ in results:
        for name in names:
            key = extract.normalize_canonical(name.canonical)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            consolidated.append(name)

    write_names_jsonl(CONSOLIDATED_PATH, consolidated)

    # ── Summary stats
    print(f"\nConsolidated dataset: {CONSOLIDATED_PATH}")
    print(
        f"Total: {len(consolidated)} unique names ({duplicates} duplicates removed)\n"
    )

    per_author: dict[str | None, int] = {}
    for spec, names, _ in results:
        per_author[spec.author] = per_author.get(spec.author, 0) + len(names)
    width = max(len(a or "—") for a in per_author) + 2
    print(f"{'author':<{width}}{'names':>7} (per source, before deduplication)")
    for author, count in sorted(
        per_author.items(), key=lambda kv: -kv[1]
    ):  # sort by count desc
        print(f"{(author or '—'):<{width}}{count:>7}")

    failed = [(s, status) for s, _n, status in results if status != "ok"]
    if failed:
        print("\nFailed sources:")
        for spec, status in failed:
            print(f"  {spec.name}: {status}")


if __name__ == "__main__":
    main()
