"""SPARQL queries for the Wikidata Query Service. This is DATA, not code.

SourceSpec(kind="sparql", target=<key>) selects a query from QUERIES.
Reference points:
  Q15632617 = humain fictif (fictional human)
  Q150      = français (French)
  Q7725634  = œuvre littéraire (literary work; les romans y sont typés
              directement, pas comme "roman" — Q8261 — en pratique)
  P1441     = présent dans l'œuvre (present in work)
  P407      = langue de l'œuvre (pour un livre ≈ langue du texte original)
  P50       = auteur (author)
"""

QUERIES: dict[str, str] = {
    # Angle a — fiction d'expression française : tout humain fictif présent
    # dans une œuvre LITTÉRAIRE (P31/P279* Q7725634, inclut romans, théâtre,
    # BD franco-belge) de langue française (P407). P407 + forme littéraire
    # exclut l'audiovisuel, les jeux et les mangas traduits ; P364/P495 ont
    # été écartés : ils ne sont pas renseignés sur les livres.
    "french_fictional_humans": """\
SELECT ?item ?work ?itemLabel ?workLabel WHERE {
  ?item wdt:P31 wd:Q15632617 ;
        wdt:P1441 ?work .
  ?work wdt:P407 wd:Q150 ;
        wdt:P31/wdt:P279* wd:Q7725634 .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "fr,en". }
}
""",
    # Angle b — ciblé : œuvres écrites par un panel d'auteurs français
    # classiques et modernes (QIDs vérifiés par sitelink frwiki).
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
