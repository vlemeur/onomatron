"""Probe candidate sources: fetch, extract, report yield.

Evidence-based source selection. Each SourceSpec is run through the
appropriate extractor and the result count is printed as a report table.
SPARQL sources are additionally compared against the Wikipédia names
(normalized canonical): the status shows new names vs duplicates,
Wikipédia being the priority source.
Re-run after editing SOURCES: disk cache makes it nearly instant.
"""

import logging

from onomatron.scraper import extract, sources, sparql_queries
from onomatron.scraper.fetch import SparqlClient, WikipediaClient
from onomatron.scraper.models import CharacterName, SourceSpec

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")


def probe_source(
    client: WikipediaClient, spec: SourceSpec, sparql_client: SparqlClient
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
                return [], "ok — catégorie vide ou inexistante"
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
    except Exception as exc:  # network failure after retries
        return [], f"échec réseau ({type(exc).__name__})"
    return [], f"kind inconnu: {spec.kind}"


def main() -> None:
    client = WikipediaClient()
    sparql_client = SparqlClient()

    report: list[tuple[SourceSpec, list[CharacterName], str]] = []
    for spec in sources.SOURCES:
        names, status = probe_source(client, spec, sparql_client)
        report.append((spec, names, status))

    # Wikipédia-priority duplicate detection: gather the normalized
    # canonicals of every non-SPARQL source, then classify SPARQL names.
    wiki_keys = {
        extract.normalize_canonical(n.canonical)
        for spec, names, _status in report
        if spec.kind != "sparql"
        for n in names
    }

    table = []
    for spec, names, status in report:
        if spec.kind == "sparql":
            dup = sum(
                1
                for n in names
                if extract.normalize_canonical(n.canonical) in wiki_keys
            )
            new = len(names) - dup
            status = (
                f"{status} — {new} nouveaux / {dup} doublons (Wikipédia prioritaire)"
            )
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
