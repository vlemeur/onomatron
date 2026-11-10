"""Extract character name candidates from wikitext and page titles."""

import logging
import re

import mwparserfromhell as mw
from pydantic import ValidationError

from onomatron.scraper.models import CharacterName

logger = logging.getLogger(__name__)

# Internal links: [[Title]] or [[Title|display]] (ignores fragments #section)
WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:\|[^\]]*)?\]\]")

# Everything after these headings is bibliography or references — links there
# are scholars, publishers and translations, not characters. ("Voir aussi" is
# deliberately absent: some list pages place it before the character lists.)
TAIL_SECTIONS_RE = re.compile(
    r"^\s*={2,}\s*(?:Notes et r[ée]f[ée]rences|R[ée]f[ée]rences|"
    r"Bibliographie|Liens externes|Articles connexes)\s*={2,}",
    re.IGNORECASE | re.MULTILINE,
)

# Proper noun sequences (1-4 words), possibly with lowercase particles
# (de, de la, d', du, des, van) — crude by design, refined in normalize.
# Single words are accepted: character page titles and list lines inside
# work sections contain many one-word names (Vautrin, Albéric, Cosette).
NAME_RE = re.compile(
    r"^[A-ZÉÈÀÂÎÔÛÇ][\wéèêàâîôûçüÉÈÀÂÎÔÛÇ\-]+"
    r"(?:\s+(?:de\s+la\s+|de\s+|d'|du\s+|des\s+|van\s+)?[A-ZÉÈÀÂÎÔÛÇ][\wéèêàâîôûçüÉÈÀÂÎÔÛÇ\-]+)*$"
)
# Stricter variant for list pages: they link far beyond characters (places,
# people, concepts), and one-word links there are almost never names
# (Damas, Graz, Caen, Russie) — require at least two words.
NAME_MULTI_RE = re.compile(
    r"^[A-ZÉÈÀÂÎÔÛÇ][\wéèêàâîôûçüÉÈÀÂÎÔÛÇ\-]+"
    r"(?:\s+(?:de\s+la\s+|de\s+|d'|du\s+|des\s+|van\s+)?[A-ZÉÈÀÂÎÔÛÇ][\wéèêàâîôûçüÉÈÀÂÎÔÛÇ\-]+)+$"
)
# Leading name at the start of a list line: 1-4 capitalized words,
# optionally with French particles ("de", "d'", "du", "des").
NAME_PREFIX_RE = re.compile(
    r"^[A-ZÉÈÀÂÎÔÛÇ][\wéèêàâîôûçüÉÈÀÂÎÔÛÇ\-]+"
    r"(?:\s+(?:de\s+|d'|du\s+|des\s+)?[A-ZÉÈÀÂÎÔÛÇ][\wéèêàâîôûçüÉÈÀÂÎÔÛÇ\-]+){0,3}"
)


# Titles of works (Le Père Goriot), categories, etc. link a lot from lists.
WORK_HINTS = ("Le ", "La ", "Les ", "L'", "Un ", "Une ", "Du ", "Des ")

# Determiners/pronouns that survive the pattern matches as bare words at the
# start of a line or title — they are never character names.
NON_NAME_WORDS = {
    "Acteurs",
    "Auteur",
    "Autres",
    "Ceci",
    "Ce",
    "Cette",
    "Ces",
    "Dernière",
    "Dernier",
    "De",
    "Dans",
    "Des",
    "Du",
    "Elle",
    "En",
    "Entre",
    "Famille",
    "Fourmis",
    "Humains",
    "Il",
    "Ils",
    "La",
    "Le",
    "Les",
    "Leur",
    "Leurs",
    "Liste",
    "Madame",
    "Monsieur",
    "Non",
    "Note",
    "Notes",
    "Nous",
    "On",
    "Parmi",
    "Personnage",
    "Personnages",
    "Présentation",
    "Première",
    "Premier",
    "Principaux",
    "Résumé",
    "Sa",
    "Sans",
    "Section",
    "Ses",
    "Synopsis",
    "Son",
    "Sources",
    "Tout",
    "Tous",
    "Un",
    "Une",
    "Vous",
}

# Names starting with these are meta-designations (Famille X, Personnages de X),
# not character names.
NON_NAME_PREFIXES = ("Famille ", "Liste ", "Personnages de ", "Types de ")

# Wikidata label service falls back to the item ID when no label exists.
QID_RE = re.compile(r"^Q\d+$")

# Contextual prefixes attached to NER aliases from running text
# ("au père Walter", "de la Maréchale") — stripped before filtering.
ALIAS_PREFIX_RE = re.compile(
    r"^(?:aux?\s+|du\s+|des\s+|de\s+la\s+|de\s+|chez\s+|[dl]')\s*", re.IGNORECASE
)


def normalize_canonical(name: str) -> str:
    """Lenient key for duplicate detection: lowercase, alphanumeric only.

    Collapses punctuation and spacing, so "Jean-Baptiste Adamsberg" and
    "Jean Baptiste Adamsberg" share a key. Accents are kept on purpose —
    we prefer missing a rare duplicate to merging distinct names.
    """
    return re.sub(r"[\W_]+", "", name.lower())


def _is_plausible_name(canonical: str) -> bool:
    """Filter surviving pattern matches that are clearly not names."""
    if canonical in NON_NAME_WORDS:
        return False
    return not canonical.startswith(NON_NAME_PREFIXES)


def extract_from_list_page(wikitext: str, source: str) -> list[CharacterName]:
    """Extract character names from a "List of characters" page.

    Strategy: the internal links' *titles* are canonical name forms.
    Disambiguation suffixes ("Vautrin (personnage de Balzac)") are stripped
    before filtering; bibliography and navigation sections are ignored;
    a first pass rejects obvious non-names (works, meta).
    """
    tail = TAIL_SECTIONS_RE.search(wikitext)
    if tail:
        wikitext = wikitext[: tail.start()]
    titles = [t.strip() for t in WIKILINK_RE.findall(wikitext)]
    seen: set[str] = set()
    names: list[CharacterName] = []
    rejected = 0
    for title in titles:
        canonical = title.split(" (")[0].strip()
        if canonical in seen or not NAME_MULTI_RE.match(canonical):
            rejected += 1
            continue
        if any(canonical.startswith(h) for h in WORK_HINTS):
            rejected += 1
            continue
        if not _is_plausible_name(canonical):
            rejected += 1
            continue
        seen.add(canonical)
        names.append(CharacterName(canonical=canonical, source=source))
    logger.info("%s: %d names extracted, %d rejected", source, len(names), rejected)
    return names


def title_to_name(
    title: str, source: str, author: str | None = None
) -> CharacterName | None:
    """Build a CharacterName from a category member page title.

    Wikipedia disambiguates character pages as "Name (personnage)" — strip the
    parenthetical. Titles that don't survive the name filter (works,
    adaptations, meta pages — common in author categories like
    "Catégorie:Arsène Lupin") are rejected: return None.
    """
    canonical = title.split(" (")[0].strip()
    if not NAME_RE.match(canonical):
        return None
    if any(canonical.startswith(h) for h in WORK_HINTS):
        return None
    if not _is_plausible_name(canonical):
        return None
    return CharacterName(canonical=canonical, source=source, author=author)


def extract_from_work_section(
    wikitext: str,
    source: str,
    author: str | None,
    section_heading: str,
) -> list[CharacterName]:
    """Extract character names from a section of a work's article.

    Unlike list pages, names in "Personnages" sections of work articles
    are often plain text at the start of list items, not wikilinks.

    Strategy: isolate the section via its heading, strip the wiki markup,
    then take the leading name-like prefix of each line.
    """
    code = mw.parse(wikitext)

    # get_sections(matches=...) filters sections by heading natively.
    # include_headings=False: keep only the section's content.
    sections = code.get_sections(
        matches=section_heading,  # plain string, the lib compiles it
        flags=re.IGNORECASE,  # case-insensitive match on headings
        include_headings=False,
    )
    if not sections:
        logger.warning("%s: no section matching %r found", source, section_heading)
        return []

    names: list[CharacterName] = []
    seen: set[str] = set()
    for section in sections:
        for line in section.strip_code().split("\n"):
            line = line.strip().lstrip("-*•").strip()
            match = NAME_PREFIX_RE.match(line)
            if not match:
                continue
            canonical = match.group(0).rstrip(" ,:")
            if not _is_plausible_name(canonical):
                continue
            if canonical in seen:
                continue
            seen.add(canonical)
            names.append(
                CharacterName(canonical=canonical, source=source, author=author)
            )

    logger.info(
        "%s: %d names extracted from section %r", source, len(names), section_heading
    )
    return names


def extract_from_sparql(
    rows: list[dict[str, str]], author: str | None = None
) -> list[CharacterName]:
    """Map SPARQL result rows to CharacterName.

    Wikidata shape: itemLabel (French label of the character) becomes
    canonical, workLabel (French label of the work) becomes source.
    Tolerant on their data: rows without a usable label — including the
    label service's Q-ID fallback — or failing internal validation are
    skipped, never fatal.
    """
    names: list[CharacterName] = []
    seen: set[str] = set()
    rejected = 0
    for row in rows:
        label = row.get("itemLabel", "").strip()
        work = row.get("workLabel", "").strip()
        if not label or not work or QID_RE.match(label):
            rejected += 1
            continue
        if label in seen or not _is_plausible_name(label):
            rejected += 1
            continue
        seen.add(label)
        try:
            names.append(CharacterName(canonical=label, source=work, author=author))
        except ValidationError:
            # strict pydantic, tolerant on their data: drop, don't crash
            rejected += 1
    logger.info("sparql: %d names extracted, %d rows rejected", len(names), rejected)
    return names


def extract_from_ner_rows(
    rows: list[dict[str, object]],
    novel_authors: dict[str, str] | None = None,
) -> list[CharacterName]:
    """Map annotated NER rows to CharacterName.

    Rows come from the sept-romans alias-resolution gold standard:
    {form, type, mentions, entity, novel}. Only PER rows carry an entity.
    Both the gold canonical (entity) and each surface alias (form) become
    candidates — the alias→entity mapping itself is not representable in
    CharacterName and belongs to the normalize phase.
    Tolerant: rows without a usable alias or entity are skipped, never
    fatal.
    """
    novel_authors = novel_authors or {}
    names: list[CharacterName] = []
    seen: set[tuple[str, str]] = set()
    rejected = 0

    def add(canonical: str, novel: str) -> None:
        nonlocal rejected
        canonical = canonical.strip()
        if not canonical or (canonical, novel) in seen:
            return
        try:
            names.append(
                CharacterName(
                    canonical=canonical,
                    source=novel,
                    author=novel_authors.get(novel),
                )
            )
            seen.add((canonical, novel))
        except ValidationError:
            rejected += 1

    for row in rows:
        if row.get("type") != "PER":
            continue
        novel = str(row.get("novel", "")).strip()
        if not novel:
            continue
        entity = str(row.get("entity", "")).strip()
        if entity and entity != "?":
            add(entity, novel)
        form = str(row.get("form", "")).strip()
        if form:
            for _ in range(2):  # strip at most two contextual prefixes
                stripped = ALIAS_PREFIX_RE.sub("", form)
                if stripped == form:
                    break
                form = stripped
            if NAME_RE.match(form) and _is_plausible_name(form):
                add(form, novel)
            else:
                rejected += 1

    logger.info(
        "ner rows: %d names extracted, %d aliases rejected", len(names), rejected
    )
    return names
