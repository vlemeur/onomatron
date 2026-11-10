"""SPARQL queries for the Wikidata Query Service. This is DATA, not code.

SourceSpec(kind="sparql", target=<key>) selects a query from QUERIES.
Reference points:
  Q15632617 = fictional human (humain fictif)
  Q150      = French (français)
  Q7725634  = literary work (œuvre littéraire); in practice novels are
              typed directly with it, unlike "roman" — Q8261
  P1441     = present in work (présent dans l'œuvre)
  P407      = language of the work (for a book ≈ language of the original
              text)
  P50       = author
"""

QUERIES: dict[str, str] = {
    # Angle a — French-language fiction: every fictional human present
    # in a LITERARY work (P31/P279* Q7725634, includes novels, theatre,
    # Franco-Belgian comics) written in French (P407). P407 + the
    # literary-form typing excludes audiovisual works, games and
    # translated manga; P364/P495 were ruled out: they are not set on
    # books.
    "french_fictional_humans": """\
SELECT ?item ?work ?itemLabel ?workLabel WHERE {
  ?item wdt:P31 wd:Q15632617 ;
        wdt:P1441 ?work .
  ?work wdt:P407 wd:Q150 ;
        wdt:P31/wdt:P279* wd:Q7725634 .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "fr,en". }
}
""",
    # Angle b — targeted: works written by a panel of classic and modern
    # French authors (QIDs verified via frwiki sitelinks).
    "authors_french_fictional_humans": """\
SELECT ?item ?work ?itemLabel ?workLabel WHERE {
  VALUES ?author {
    wd:Q9711     # Honoré de Balzac
    wd:Q38337    # Alexandre Dumas
    wd:Q535      # Victor Hugo
    wd:Q504      # Émile Zola
    wd:Q7199     # Marcel Proust
    wd:Q34670    # Albert Camus
    wd:Q9327     # Guy de Maupassant
    wd:Q43444    # Gustave Flaubert
    wd:Q502      # Stendhal
    wd:Q33977    # Jules Verne
    wd:Q23441    # Octave Mirbeau
    wd:Q687      # Molière
    wd:Q12054    # Bernard Werber
    wd:Q2829704  # Alain Damasio
    wd:Q3219246  # Laurent Généfort
    wd:Q3168813  # Jean-Philippe Jaworski
  }
  ?item wdt:P31 wd:Q15632617 ;
        wdt:P1441 ?work .
  ?work wdt:P50 ?author .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "fr,en". }
}
""",
}
