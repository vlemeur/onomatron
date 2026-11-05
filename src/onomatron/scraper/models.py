"""Data models for the scraper pipeline."""

from pydantic import BaseModel, Field


class CharacterName(BaseModel):
    """A character name candidate extracted from a source."""

    canonical: str = Field(min_length=2, max_length=80)
    source: (
        str  # where it came from, e.g. "Liste des personnages de La Comédie humaine"
    )
    author: str | None = None


class SourceSpec(BaseModel):
    """A scrape target. Adding a new author/era = adding a line here."""

    name: str
    kind: str = Field(pattern="^(list_page|category|work_page)$")
    target: str
    section: str | None = None  # work_page only: heading to target
    author: str | None = None
