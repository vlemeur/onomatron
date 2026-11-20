"""Tests for the Phase 2 normalization (pure transformation, no network).

The cases below are the documented behaviors of clean_name /
sanitize_for_model — the ones the scraper paid for in hours — plus a few
properties that must hold for any input.
"""

import unicodedata

import pytest

from onomatron.scraper.normalize import (
    clean_name,
    dedupe_names,
    is_name_like,
    sanitize_for_model,
)


class TestCleanName:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Rodrigue (Rodrigo)", "Rodrigue"),  # parenthetical qualifier
            ("Louis XIV", "Louis"),  # roman ordinal dropped
            ("Henri IV", "Henri"),
            ("Muskar I", "Muskar"),  # single "I" counts as an ordinal
            ("Monsieur de Valmont", "Valmont"),  # address + particle
            ("Général Hugo", "Hugo"),  # title dropped
            ("Madeleine Souvray, marquise Sablé", "Madeleine Souvray"),  # comma
            ("Marquise Mole et Mathilde Mole", "Mole"),  # conjunction + title
            ("Baron L...", "L"),  # editorial debris trimmed at the edge
            ("  Cosette  ", "Cosette"),  # whitespace collapsed
        ],
    )
    def test_documented_cases(self, raw: str, expected: str) -> None:
        assert clean_name(raw) == expected

    @pytest.mark.parametrize(
        "junk",
        [
            "The Duke",
            "Old Man",
            "2nd Peasant",
            "First gravedigger",
            "The Factory Girl",
            "Un tavernier",
            "A hunter",
            "Les Thénardier",
            "L'officier espagnol",
        ],
    )
    def test_junk_markers_kill_the_whole_candidate(self, junk: str) -> None:
        assert clean_name(junk) == ""

    def test_stray_open_paren_cuts_the_rest(self) -> None:
        assert clean_name("Cosette (née") == "Cosette"


class TestIsNameLike:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("", False),
            ("L", False),
            ("L...", False),  # one letter, the dots are not letters
            ("Ab", True),
            ("Éléonore", True),
        ],
    )
    def test_at_least_two_letters(self, name: str, expected: bool) -> None:
        assert is_name_like(name) is expected


class TestDedupeNames:
    def test_first_occurrence_wins(self) -> None:
        unique, dropped = dedupe_names(["Henri", "henri", "Henri"])
        assert unique == ["Henri"]
        assert dropped == 2

    def test_key_ignores_case_and_punctuation(self) -> None:
        unique, dropped = dedupe_names(
            ["Jean-Baptiste", "Jean Baptiste", "jeanbaptiste"]
        )
        assert unique == ["Jean-Baptiste"]
        assert dropped == 2

    def test_accents_are_significant(self) -> None:
        # normalize_canonical keeps accents on purpose: missing a rare
        # duplicate is preferred over merging distinct names
        unique, dropped = dedupe_names(["Éléonore", "Eleonore"])
        assert len(unique) == 2
        assert dropped == 0


class TestSanitizeForModel:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Cosette", ["cosette"]),
            ("Éléonore", ["éléonore"]),  # French accents are kept
            ("Straße", ["strasse"]),  # ß -> ss
            ("Œdipe", ["oedipe"]),  # œ -> oe
            ("d'Artagnan", ["artagnan"]),  # apostrophe -> space, 1-letter part dropped
            ("Nom Prénom", ["nom", "prénom"]),  # full name splits into two names
            ("R2D2", []),  # digits drop the whole candidate
            ("Jean-Paul", []),  # hyphens drop the whole candidate
            ("警告", []),  # non-latin script drops the whole candidate
        ],
    )
    def test_documented_cases(self, raw: str, expected: list[str]) -> None:
        assert sanitize_for_model(raw) == expected

    def test_output_alphabet_is_latin_letters_only(self, names: list[str]) -> None:
        for name in names:
            for part in sanitize_for_model(name):
                assert part == part.lower()
                assert sum(ch.isalpha() for ch in part) >= 2
                assert all(unicodedata.name(ch).startswith("LATIN") for ch in part), (
                    f"non-latin letter in {part!r}"
                )
