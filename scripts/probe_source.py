"""Probe candidate sources: fetch, extract, report yield.

Evidence-based source selection. Each SourceSpec is run through the
appropriate extractor and the result count is printed as a report table.
SPARQL sources are additionally compared against the Wikipedia names
(normalized canonical): the status shows new names vs duplicates,
Wikipedia being the priority source.
Re-run after editing SOURCES: disk cache makes it nearly instant.
"""

import logging

from onomatron.scraper import extract, hf_datasets, sources, sparql_queries
from onomatron.scraper.fetch import (
    HuggingFaceClient,
    SparqlClient,
    WikipediaClient,
)
from onomatron.scraper.models import CharacterName, SourceSpec

# Kinds whose names are compared against the Wikipedia corpus for the
# "new / duplicates" status: everything not sourced from Wikipedia.
EXTERNAL_KINDS = ("sparql", "hf_dataset")

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")


def probe_source(
    client: WikipediaClient,
    spec: SourceSpec,
    sparql_client: SparqlClient,
    hf_client: HuggingFaceClient,
) -> tuple[list[CharacterName], str]:
    """Return (names, status) for one source. Never raises: a broken
    source is information, not an error — report it and move on."""
    names: list[CharacterName] = []
    try:
        if spec.kind == "category":
            # category members are listed via their own API: never fetch
            # wikitext for them (Catégorie:X targets have no article page).
            pages, _subcats = client.get_category_members(spec.target)
            seen: set[str] = set()
            for p in pages:
                name = extract.title_to_name(p, source=spec.target, author=spec.author)
                if name is None or name.canonical in seen:
                    continue
                seen.add(name.canonical)
                names.append(name)
            if not pages:
                return [], "ok — category empty or nonexistent"
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
    except Exception as exc:  # network failure after retries
        return [], f"network failure ({type(exc).__name__})"
    return [], f"unknown kind: {spec.kind}"


def main() -> None:
    client = WikipediaClient()
    sparql_client = SparqlClient()
    hf_client = HuggingFaceClient()

    report: list[tuple[SourceSpec, list[CharacterName], str]] = []
    for spec in sources.SOURCES:
        names, status = probe_source(client, spec, sparql_client, hf_client)
        report.append((spec, names, status))

    # Wikipedia-priority duplicate detection: gather the normalized
    # canonicals of every Wikipedia-sourced source, then classify the
    # external sources' names.
    wiki_keys = {
        extract.normalize_canonical(n.canonical)
        for spec, names, _status in report
        if spec.kind not in EXTERNAL_KINDS
        for n in names
    }

    table = []
    for spec, names, status in report:
        if spec.kind in EXTERNAL_KINDS:
            dup = sum(
                1
                for n in names
                if extract.normalize_canonical(n.canonical) in wiki_keys
            )
            new = len(names) - dup
            status = f"{status} — {new} new / {dup} duplicates (Wikipedia has priority)"
        table.append((spec.name, spec.kind, spec.author or "—", len(names), status))

    # Pretty-print a fixed-width table
    width = max(len(r[0]) for r in table) + 2
    print(f"{'source':<{width}}{'kind':<12}{'author':<22}{'names':>6}  status")
    print("-" * (width + 60))
    total = 0
    for name, kind, author, count, status in sorted(table, key=lambda r: -r[3]):
        total += count
        print(f"{name:<{width}}{kind:<12}{author[:20]:<22}{count:>6}  {status}")
    print("-" * (width + 60))
    print(f"TOTAL: {total} candidate names across {len(table)} sources")


if __name__ == "__main__":
    main()
