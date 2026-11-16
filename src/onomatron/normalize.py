"""Phase 2 — normalize the scraped name candidates.

Pure transformation, no network. clean_name() first strips parenthetical
qualifiers ("Rodrigue (Rodrigo)" -> "Rodrigue"), then kills the whole
candidate if it carries an English-role / indefinite marker (the, old,
first, girl, a clerk kind of "A hunter", un, les...): those are opera
role lists, not names. Otherwise it strips, whole-word and
case-insensitive:
- addresses and their abbreviations (monsieur, madame, mme...),
- particles (de, la, le, d', son...),
- noble / military / religious titles (comte, duc, général,
  dr, colonel...),
- meta words (personnage, famille),
- country names (France, Angleterre),
and drops uppercase roman ordinals anywhere in the name (Louis XIV ->
Louis). Designations glued together with a comma or a conjunction keep
only their first segment. A cleaned form with fewer than two letters is
not a name: is_name_like() tells the caller to drop it. dedupe_names()
then merges the duplicates the cleanup created (Henri IV / Henri),
keeping the first occurrence, same key as the Phase 1 consolidation.
"""

import re

from onomatron.scraper.extract import normalize_canonical

# Parenthetical qualifiers, mostly Wikidata disambiguation
# ("Rodrigue (Rodrigo)"). Removed with their content, whole candidate
# kept. A stray unterminated "(" cuts everything after it.
PARENTHETICAL_RE = re.compile(r"\s*\([^)]*\)")
STRAY_PAREN_RE = re.compile(r"\s*\(+[^)]*$")

# Lowercase whole-word markers of "not a name at all": English opera
# role lists (The Duke, 2nd Peasant, First gravedigger, The Factory
# Girl), indefinite designations (Un tavernier, A hunter) and family
# collectives (Les Thénardier). One marker anywhere kills the whole
# candidate. Also matches the elided article "l'" (L'officer espagnol);
# unlike the stripped particle "d'", no real name survives it here.
JUNK_MARKER_RE = re.compile(
    r"\b(?:the|old|girl|clerk|first|second|third|fourth|fifth|"
    r"1st|2nd|3rd|4th|5th|un|les|a)\b",
    re.IGNORECASE,
)
L_APOSTROPHE_RE = re.compile(r"(?<![\w'-])l'", re.IGNORECASE)

# Lowercase whole-word tokens removed wherever they appear in a name.
# The core list comes from the manual dataset pass; same-family entries
# observed in the data are included: elided particles (du, des), the rest
# of the noble ladder (marquis, comtesse, vicomte...), their English
# equivalents (lady, lord, sir, miss), and other military / religious /
# professional titles in the same spirit as Général and Pape.
STOP_TOKENS = frozenset(
    {
        # addresses and abbreviations
        "monsieur",
        "madame",
        "mr",
        "mme",
        "famille",
        "miss",
        "mlle",
        "mademoiselle",
        "lady",
        "lord",
        "sir",
        # particles
        "de",
        "la",
        "le",
        "du",
        "des",
        "dit",
        "son",
        "sa",
        "votre",
        # noble / military / religious / professional titles
        "roi",
        "reine",
        "duc",
        "duchesse",
        "baron",
        "baronne",
        "sieur",
        "dame",
        "comte",
        "compte",
        "comtesse",
        "marquis",
        "marquise",
        "vicomte",
        "vicomtesse",
        "chevalier",
        "prince",
        "princesse",
        "empereur",
        "général",
        "general",
        "sire",
        "ponce",
        "pape",
        "abbé",
        "abbe",
        "cardinal",
        "curé",
        "cure",
        "pasteur",
        "révérend",
        "reverend",
        "capitaine",
        "lieutenant",
        "sergent",
        "caporal",
        "docteur",
        "professeur",
        "maître",
        "maitre",
        "me",
        "dr",
        "gen",
        "colonel",
        # relatives used as designations
        "père",
        "pere",
        "mère",
        "mere",
        # honorific phrases (sa grâce, son altesse...) handled token-wise
        "grâce",
        "grace",
        "sainteté",
        "saintete",
        "saint",
        "altesse",
        "éminence",
        "eminence",
        # meta words
        "personnage",
        "personage",
        # countries
        "angleterre",
        "france",
    }
)

STOP_TOKENS_RE = re.compile(
    r"\b(?:" + "|".join(sorted(STOP_TOKENS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# Elided particle "d'" (d'Aiglemont): the apostrophe is part of the token,
# so \b word boundaries alone cannot catch it. Case-insensitive.
D_APOSTROPHE_RE = re.compile(r"(?<![\w'-])d'", re.IGNORECASE)

# Uppercase roman numerals (ordinals: Louis XIV, Henri IV). The pattern
# is deliberately strict so real uppercased words made only of the
# letters IVXLCDM (LI in "Lili" is lowercase, DIC does not parse) stay.
# Single "I" counts: "Muskar I" must merge into "Muskar".
ROMAN_NUMERAL_RE = re.compile(
    r"M{0,3}(?:CM|CD|D?C{0,3})?(?:XC|XL|L?X{0,3})?(?:IX|IV|V?I{0,3})?"
)

# Stray separators and editorial junk left on the edges after the
# token removals (Baron L... -> " L...", Dr. X -> ". X").
EDGE_JUNK_RE = re.compile(r"^[\s\-''…«».,;:]+|[\s\-''…«».,;:]+$")

# Some candidates are two designations glued together with a comma or a
# conjunction ("Madeleine Souvray, marquise Sablé", "Marquise Mole et
# Mathilde Mole"): keep only the first one.
SEGMENT_SPLIT_RE = re.compile(r"\s*[,;]\s*|\s+(?:et|and)\s+", re.IGNORECASE)


def _is_roman_numeral(token: str) -> bool:
    return token.isupper() and bool(ROMAN_NUMERAL_RE.fullmatch(token))


def clean_name(name: str) -> str:
    """Return the cleaned form of one name candidate.

    Parenthetical qualifiers go first, then a candidate carrying a
    junk marker (English role, indefinite designation) is killed
    whole. Otherwise stop tokens and roman ordinals are removed
    anywhere in the string (whole words only), then whitespace is
    collapsed and the edges are trimmed of leftover punctuation. May
    return "" for a candidate that was nothing but honorifics.
    """
    cleaned = PARENTHETICAL_RE.sub("", name)
    cleaned = STRAY_PAREN_RE.sub("", cleaned)
    if JUNK_MARKER_RE.search(cleaned) or L_APOSTROPHE_RE.search(cleaned):
        return ""
    cleaned = SEGMENT_SPLIT_RE.split(cleaned)[0]
    cleaned = STOP_TOKENS_RE.sub(" ", cleaned)
    cleaned = D_APOSTROPHE_RE.sub(" ", cleaned)
    tokens = [t for t in cleaned.split() if not _is_roman_numeral(t)]
    cleaned = " ".join(tokens)
    return EDGE_JUNK_RE.sub("", cleaned)


def is_name_like(name: str) -> bool:
    """True if the cleaned form still looks like a proper name: at
    least two letters. Drops honorific-only leftovers ("Sa") and
    editorial debris ("L...")."""
    return sum(c.isalpha() for c in name) >= 2


def dedupe_names(names: list[str]) -> tuple[list[str], int]:
    """Deduplicate cleaned names, first occurrence wins. Returns
    (unique names, number of duplicates dropped)."""
    seen: set[str] = set()
    unique: list[str] = []
    for name in names:
        key = normalize_canonical(name)
        if key in seen:
            continue
        seen.add(key)
        unique.append(name)
    return unique, len(names) - len(unique)
