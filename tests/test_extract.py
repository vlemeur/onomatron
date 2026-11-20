"""Tests for the pure extraction layer (no network) and the pydantic models."""

import pytest
from pydantic import ValidationError

from onomatron.scraper.extract import (
    extract_from_list_page,
    extract_from_raw_lines,
    normalize_canonical,
    title_to_name,
)
from onomatron.scraper.models import CharacterName, SourceSpec


class TestNormalizeCanonical:
    def test_collapses_case_spacing_and_punctuation(self) -> None:
        key = "jeanbaptisteadamsberg"
        assert normalize_canonical("Jean-Baptiste Adamsberg") == key
        assert normalize_canonical("Jean Baptiste Adamsberg") == key

    def test_keeps_accents(self) -> None:
        # accents are kept on purpose: prefer missing a rare duplicate
        # over merging distinct names
        assert normalize_canonical("Éléonore") != normalize_canonical("Eleonore")


class TestExtractFromRawLines:
    def test_skips_blank_and_duplicate_lines(self) -> None:
        names = extract_from_raw_lines(
            ["Cosette", "", "   ", "Cosette", " Javert "], source="local.txt"
        )
        assert [n.canonical for n in names] == ["Cosette", "Javert"]

    def test_skips_lines_failing_model_validation(self) -> None:
        names = extract_from_raw_lines(["A", "x" * 81, "Cosette"], source="local.txt")
        assert [n.canonical for n in names] == ["Cosette"]

    def test_author_is_propagated(self) -> None:
        (name,) = extract_from_raw_lines(["Cosette"], source="local.txt", author="Hugo")
        assert name.author == "Hugo"
        assert name.source == "local.txt"


LIST_PAGE = """
== Personnages ==
* [[Eugène de Rastignac (personnage)]]
* [[Vautrin (personnage de Balzac)|Vautrin]]
* [[Le Père Goriot]]
* [[Cosette]]

== Bibliographie ==
* [[Honoré de Balzac]]
"""


class TestExtractFromListPage:
    def test_extracts_multiword_wikilink_titles(self) -> None:
        names = extract_from_list_page(LIST_PAGE, source="Liste du Père Goriot")
        canonicals = [n.canonical for n in names]
        assert "Eugène de Rastignac" in canonicals  # disambiguation stripped

    def test_rejects_work_titles_single_words_and_bibliography(self) -> None:
        names = extract_from_list_page(LIST_PAGE, source="Liste du Père Goriot")
        canonicals = [n.canonical for n in names]
        assert "Le Père Goriot" not in canonicals  # work hint
        assert "Cosette" not in canonicals  # single word: not a list-page name
        assert "Vautrin" not in canonicals  # single word after stripping
        assert "Honoré de Balzac" not in canonicals  # bibliography tail cut


class TestTitleToName:
    def test_strips_disambiguation_and_keeps_author(self) -> None:
        name = title_to_name(
            "Cosette (personnage des Misérables)", source="catégorie", author="Hugo"
        )
        assert name is not None
        assert name.canonical == "Cosette"
        assert name.author == "Hugo"

    @pytest.mark.parametrize(
        "title",
        [
            "La Comédie humaine",  # work hint
            "Un Américain à Paris",  # work hint
            "Personnages de Balzac",  # meta prefix
            "Liste des personnages",  # meta prefix
        ],
    )
    def test_rejects_non_names(self, title: str) -> None:
        assert title_to_name(title, source="catégorie") is None


class TestPydanticModels:
    def test_character_name_requires_two_chars(self) -> None:
        with pytest.raises(ValidationError):
            CharacterName(canonical="A", source="s")

    def test_source_spec_rejects_unknown_kind(self) -> None:
        with pytest.raises(ValidationError):
            SourceSpec(name="x", kind="telepathy", target="t")

    def test_source_spec_accepts_all_documented_kinds(self) -> None:
        for kind in (
            "list_page",
            "category",
            "work_page",
            "sparql",
            "hf_dataset",
            "raw_file",
        ):
            assert SourceSpec(name="x", kind=kind, target="t").kind == kind
