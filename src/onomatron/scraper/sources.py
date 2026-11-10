"""Configuration of scrape targets. This is DATA, not code.

Sources removed after probing (exact titles verified via the MediaWiki API;
none of these articles lists characters on the French Wikipedia):
  - rocambole:         Catégorie:Rocambole contains works and adaptations,
                       not character pages; the character article has no
                       "Personnages" section.
  - hommes_bonne_volonte: "Les Hommes de bonne volonté" has no character list.
  - princesse_cleves:  "La Princesse de Clèves" article analyses the novel,
                       no character section.
  - figaro:            "Le Barbier de Séville" has no character section; its
                       "Distribution" is an actors table (markup rows), not a
                       name list.
  - nana, une_vie:     no character section in the articles.
  - a_rebours:         "À rebours" has no character section.
  - diaboliques:       "Les Diaboliques (nouvelles)" has per-story sections,
                       no character list.
  - voyage_nuit:       "Voyage au bout de la nuit" has no character section.
  - nestor_burma:      article covers the series, no character list.
  - au_revoir_la_haut: article has no character section.
  - adamsberg:         "Jean-Baptiste Adamsberg" describes the entourage in
                       prose, no character list.
  - zone_dehors:       "La Zone du Dehors" has no character section.
  - gagner_la_guerre:  "Gagner la guerre" has no character section.
"""

from pathlib import Path

from onomatron.scraper.models import SourceSpec

SOURCES: list[SourceSpec] = [
    SourceSpec(
        name="comedie_humaine",
        kind="list_page",
        target="Liste des personnages de La Comédie humaine",
        author="Honoré de Balzac",
    ),
    SourceSpec(
        name="balzac_personnages",
        kind="category",
        target="Personnage d'Honoré de Balzac",
        author="Honoré de Balzac",
    ),
    SourceSpec(
        name="comedie_humaine_types",
        kind="list_page",
        target="Types de personnages de La Comédie humaine",
        author="Honoré de Balzac",
    ),
    SourceSpec(
        name="verne_personnages",
        kind="list_page",
        target="Liste des personnages de Jules Verne",
        author="Jules Verne",
    ),
    SourceSpec(
        name="verne_personnages_cat",
        kind="category",
        target="Personnage de Jules Verne",
        author="Jules Verne",
    ),
    SourceSpec(
        name="vingt_mille_lieues",
        kind="category",
        target="Personnage de Vingt Mille Lieues sous les mers",
        author="Jules Verne",
    ),
    SourceSpec(
        name="mousquetaires",
        kind="work_page",
        target="Les Trois Mousquetaires",
        section="Personnages",
        author="Alexandre Dumas",
    ),
    SourceSpec(
        name="monte_cristo",
        kind="work_page",
        target="Le Comte de Monte-Cristo",
        section="Personnages",
        author="Alexandre Dumas",
    ),
    SourceSpec(
        name="dumas_personnages",
        kind="category",
        target="Personnage d'Alexandre Dumas",
        author="Alexandre Dumas",
    ),
    SourceSpec(
        name="recherche_temps_perdu",
        kind="list_page",
        target="Liste des personnages d'À la recherche du temps perdu",
        author="Marcel Proust",
    ),
    SourceSpec(
        name="recherche_temps_perdu_cat",
        kind="category",
        target="Personnage d'À la recherche du temps perdu",
        author="Marcel Proust",
    ),
    SourceSpec(
        name="misérables",
        kind="category",
        target="Personnage des Misérables",
        author="Victor Hugo",
    ),
    SourceSpec(
        name="miserables_section",
        kind="work_page",
        target="Les Misérables",
        section="Les personnages",
        author="Victor Hugo",
    ),
    SourceSpec(
        name="notre_dame_paris",
        kind="category",
        target="Personnage de Notre-Dame de Paris (Hugo)",
        author="Victor Hugo",
    ),
    SourceSpec(
        name="notre_dame_paris_roman",
        kind="work_page",
        target="Notre-Dame de Paris (roman)",
        section="Personnages principaux",
        author="Victor Hugo",
    ),
    SourceSpec(
        name="vie_mode_emploi",
        kind="list_page",
        target="Liste des personnages de La Vie mode d'emploi",
        author="Georges Perec",
    ),
    SourceSpec(
        name="fantomas",
        kind="category",
        target="Fantômas",
        author="Pierre Souvestre et Marcel Allain",
    ),
    SourceSpec(
        name="arsene_lupin",
        kind="category",
        target="Arsène Lupin",
        author="Maurice Leblanc",
    ),
    SourceSpec(
        name="rouletabille",
        kind="work_page",
        target="Le Mystère de la chambre jaune",
        section="Personnages",
        author="Gaston Leroux",
    ),
    SourceSpec(
        name="pardaillan",
        kind="work_page",
        target="Les Pardaillan",
        section="Les personnages",
        author="Michel Zévaco",
    ),
    SourceSpec(
        name="liaisons_dangereuses",
        kind="category",
        target="Personnage des Liaisons dangereuses",
        author="Pierre Choderlos de Laclos",
    ),
    SourceSpec(
        name="mirbeau",
        kind="category",
        target="Personnage d'Octave Mirbeau",
        author="Octave Mirbeau",
    ),
    SourceSpec(
        name="moliere",
        kind="category",
        target="Personnage d'une pièce de théâtre de Molière",
        author="Molière",
    ),
    # ── Tier 2: canon par sections (work_page) ──
    SourceSpec(
        name="manon_lescaut",
        kind="work_page",
        target="Manon Lescaut",
        section="Personnages",
        author="Antoine François Prévost",
    ),
    SourceSpec(
        name="germinal",
        kind="work_page",
        target="Germinal (roman)",
        section="Personnages",
        author="Émile Zola",
    ),
    SourceSpec(
        name="assommoir",
        kind="work_page",
        target="L'Assommoir",
        section="Personnages",
        author="Émile Zola",
    ),
    SourceSpec(
        name="therese_raquin",
        kind="work_page",
        target="Thérèse Raquin",
        section="Personnages",
        author="Émile Zola",
    ),
    SourceSpec(
        name="madame_bovary",
        kind="work_page",
        target="Madame Bovary",
        section="Personnages",
        author="Gustave Flaubert",
    ),
    SourceSpec(
        name="salambo",
        kind="work_page",
        target="Salammbô",
        section="Personnages",
        author="Gustave Flaubert",
    ),
    SourceSpec(
        name="rouge_et_noir",
        kind="work_page",
        target="Le Rouge et le Noir",
        section="Personnages",
        author="Stendhal",
    ),
    SourceSpec(
        name="chartreuse_parme",
        kind="work_page",
        target="La Chartreuse de Parme",
        section="Personnages",
        author="Stendhal",
    ),
    SourceSpec(
        name="bel_ami",
        kind="work_page",
        target="Bel-Ami",
        section="Personnages",
        author="Guy de Maupassant",
    ),
    SourceSpec(
        name="mysteres_paris",
        kind="work_page",
        target="Les Mystères de Paris",
        section="Personnages",
        author="Eugène Sue",
    ),
    SourceSpec(
        name="grand_meaulnes",
        kind="work_page",
        target="Le Grand Meaulnes",
        section="Personnages",
        author="Alain-Fournier",
    ),
    SourceSpec(
        name="la_peste",
        kind="work_page",
        target="La Peste",
        section="Personnages",
        author="Albert Camus",
    ),
    SourceSpec(
        name="etranger",
        kind="work_page",
        target="L'Étranger",
        section="Personnages",
        author="Albert Camus",
    ),
    SourceSpec(
        name="faux_monnayeurs",
        kind="work_page",
        target="Les Faux-monnayeurs",
        section="Les personnages|Les familles",
        author="André Gide",
    ),
    SourceSpec(
        name="ecume_jours",
        kind="work_page",
        target="L'Écume des jours",
        section="Personnages",
        author="Boris Vian",
    ),
    SourceSpec(
        name="zazie",
        kind="work_page",
        target="Zazie dans le métro",
        section="Personnages",
        author="Raymond Queneau",
    ),
    SourceSpec(
        name="horde_contrevent",
        kind="work_page",
        target="La Horde du Contrevent",
        section="Personnages",
        author="Alain Damasio",
    ),
    SourceSpec(
        name="furtifs",
        kind="work_page",
        target="Les Furtifs",
        section="Personnages",
        author="Alain Damasio",
    ),
    SourceSpec(
        name="omale",
        kind="work_page",
        target="Omale",
        section="Personnages principaux|Autres personnages",
        author="Laurent Généfort",
    ),
    SourceSpec(
        name="ravage",
        kind="work_page",
        target="Ravage (roman)",
        section="Personnages",
        author="René Barjavel",
    ),
    SourceSpec(
        name="nuit_des_temps",
        kind="work_page",
        target="La Nuit des temps",
        section="Personnages",
        author="René Barjavel",
    ),
    SourceSpec(
        name="planete_singes",
        kind="work_page",
        target="La Planète des singes",
        section="Personnages",
        author="Pierre Boulle",
    ),
    SourceSpec(
        name="fourmis",
        kind="work_page",
        target="Les Fourmis (Werber)",
        section="Personnages",
        author="Bernard Werber",
    ),
    # ── Tier 3: Wikidata (SPARQL) ──
    # Placed last: consolidation keeps the FIRST occurrence of a normalized
    # canonical, so Wikipédia stays the priority source of every duplicate.
    # Angle a is broad and noisy by design (works in French, incl. foreign
    # literature documented in French); angle b is the targeted panel of
    # French authors. Both are trimmed later — raw counts remain auditable
    # in the per-source JSONL files.
    SourceSpec(
        name="wikidata_french_lit",
        kind="sparql",
        target="french_fictional_humans",
    ),
    SourceSpec(
        name="wikidata_authors",
        kind="sparql",
        target="authors_french_fictional_humans",
    ),
    # ── Tier 4: Hugging Face — gold NER annotation (ObTIC/Sorbonne) ──
    # Same priority rule as Wikidata: placed last so Wikipédia wins on
    # duplicates. Biggest density gain: ~1 600 gold canonical entities on
    # 7 novels already in the corpus (e.g. 318 for Le Rouge et le Noir vs
    # 12 on Wikipédia); the surface aliases only exist here. Please cite
    # Maurel et al., DH 2025 on reuse.
    SourceSpec(
        name="sept_romans_alias",
        kind="hf_dataset",
        target="seven_novels_alias_resolution",
    ),
]


def output_path(spec: SourceSpec, root: str | Path = "data/interim") -> Path:
    return Path(root) / f"{spec.name}.jsonl"
