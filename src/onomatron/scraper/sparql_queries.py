"""SPARQL queries for the Wikidata Query Service. This is DATA, not code.

SourceSpec(kind="sparql", target=<key>) selects a query from QUERIES.
Reference points:
  Q15632617 = humain fictif (fictional human)
  Q150      = français (French)
  P1441     = présent dans l'œuvre (present in work)
  P407      = langue de l'œuvre (language of work or name)
  P50       = auteur (author)
"""

QUERIES: dict[str, str] = {
    # Angle a — large : tout humain fictif présent dans une œuvre de langue
    # française. Bruité par conception (œuvres étrangères documentées en
    # français incluses) ; le tri fin appartient à la phase normalize.
    "french_fictional_humans": """\
SELECT ?item ?work ?itemLabel ?workLabel WHERE {
  ?item wdt:P31 wd:Q15632617 ;
        wdt:P1441 ?work .
  ?work wdt:P407 wd:Q150 .
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
