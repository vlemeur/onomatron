"""Extract character name candidates from wikitext and page titles."""

import logging
import re

from onomatron.scraper.models import CharacterName

logger = logging.getLogger(__name__)

# Internal links: [[Title]] or [[Title|display]] (ignores fragments #section)
WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:\|[^\]]*)?\]\]")

# Multi-word proper noun sequences, possibly with lowercase particles
# ("de", "d'", "du", "la", "le") — crude by design, refined in normalize.
NAME_RE = re.compile(
    r"^[A-ZÉÈÀÂÎÔÛÇ][\wéèêàâîôûçüÉÈÀÂÎÔÛÇ\-]+"
    r"(?:\s+(?:de\s+|d'|du\s+|des\s+)?[A-ZÉÈÀÂÎÔÛÇ][\wéèêàâîôûçüÉÈÀÂÎÔÛÇ\-]+)+$"
)

# Titles of works (Le Père Goriot), categories, etc. link a lot from lists.
WORK_HINTS = ("Le ", "La ", "Les ", "L'", "Un ", "Une ", "Du ", "Des ")


def extract_from_list_page(wikitext: str, source: str) -> list[CharacterName]:
    """Extract character names from a "List of characters" page.

    Strategy: the internal links' *titles* are canonical name forms.
    A first pass filters obvious non-names (works, single words).
    """
    titles = [t.strip() for t in WIKILINK_RE.findall(wikitext)]
    seen: set[str] = set()
    names: list[CharacterName] = []
    rejected = 0
    for title in titles:
        if title in seen or not NAME_RE.match(title):
            rejected += 1
            continue
        if any(title.startswith(h) for h in WORK_HINTS):
            rejected += 1
            continue
        seen.add(title)
        names.append(CharacterName(canonical=title, source=source))
    logger.info("%s: %d names extracted, %d rejected", source, len(names), rejected)
    return names


def title_to_name(title: str, source: str, author: str | None = None) -> CharacterName:
    """Build a CharacterName from a category member page title.

    Wikipedia disambiguates character pages as "Name (personnage)" — strip it.
    """
    canonical = title.split(" (")[0].strip()
    return CharacterName(canonical=canonical, source=source, author=author)
