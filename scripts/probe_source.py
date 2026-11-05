"""Probe candidate sources: fetch, extract, report yield.

Evidence-based source selection. Each SourceSpec is run through the
appropriate extractor and the result count is printed as a report table.
Re-run after editing SOURCES: disk cache makes it nearly instant.
"""

import logging

from onomatron.scraper import extract, sources
from onomatron.scraper.fetch import WikipediaClient
from onomatron.scraper.models import SourceSpec

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")


def probe_source(client: WikipediaClient, spec: SourceSpec) -> tuple[int, str]:
    """Return (name_count, status) for one source. Never raises: a broken
    source is information, not an error — report it and move on."""
    try:
        # category members are listed via their own API: never fetch wikitext
        # for them (Catégorie:X targets have no article page).
        if spec.kind == "category":
            pages, _subcats = client.get_category_members(spec.target)
            names = []
            seen: set[str] = set()
            for p in pages:
                name = extract.title_to_name(p, source=spec.target, author=spec.author)
                if name is None or name.canonical in seen:
                    continue
                seen.add(name.canonical)
                names.append(name)
            if not pages:
                return 0, "ok — catégorie vide ou inexistante"
            return len(names), "ok"
    except Exception as exc:  # network failure after retries
        return 0, f"échec réseau ({type(exc).__name__})"

    try:
        wikitext = client.get_wikitext(spec.target)
    except ValueError as exc:
        return 0, f"page introuvable ({exc})"
    except Exception as exc:  # network failure after retries
        return 0, f"échec réseau ({type(exc).__name__})"

    if spec.kind == "list_page":
        names = extract.extract_from_list_page(wikitext, source=spec.target)
    elif spec.kind == "work_page":
        if spec.section is None:
            return 0, "work_page sans section"
        names = extract.extract_from_work_section(
            wikitext,
            source=spec.target,
            author=spec.author,
            section_heading=spec.section,
        )
    else:
        return 0, f"kind inconnu: {spec.kind}"

    return len(names), "ok"


def main() -> None:
    client = WikipediaClient()

    report = []
    for spec in sources.SOURCES:
        count, status = probe_source(client, spec)
        report.append((spec.name, spec.kind, spec.author or "—", count, status))

    # Pretty-print a fixed-width table
    width = max(len(r[0]) for r in report) + 2
    print(f"{'source':<{width}}{'kind':<12}{'author':<22}{'names':>6}  status")
    print("-" * (width + 48))
    total = 0
    for name, kind, author, count, status in sorted(report, key=lambda r: -r[3]):
        total += count
        print(f"{name:<{width}}{kind:<12}{author[:20]:<22}{count:>6}  {status}")
    print("-" * (width + 48))
    print(f"TOTAL: {total} candidate names across {len(report)} sources")


if __name__ == "__main__":
    main()
