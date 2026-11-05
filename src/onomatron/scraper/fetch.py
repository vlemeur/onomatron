"""MediaWiki API client with disk cache, retries and polite pacing."""

import hashlib
import logging
import time
from pathlib import Path

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

API_URL = "https://fr.wikipedia.org/w/api.php"
REQUEST_DELAY = 0.5  # seconds between API calls — be a good citizen

USER_AGENT = "onomatron/0.1 (https://github.com/vlemeur/onomatron; educational project)"


class WikipediaClient:
    """Fetches wikitext from the French Wikipedia, caching on disk.

    Each page's wikitext is stored under cache_dir, so the pipeline can be
    re-run (parsing, normalizing) without hitting the API again.
    """

    def __init__(self, cache_dir: Path | str = "data/raw") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._session.headers["User-Agent"] = USER_AGENT

    # -- public API ---------------------------------------------------

    def get_wikitext(self, title: str) -> str:
        """Return the wikitext of a page, from cache if available."""
        cache_path = self._cache_path(title)
        if cache_path.exists():
            logger.debug("cache hit: %s", title)
            return cache_path.read_text(encoding="utf-8")

        logger.info("fetching: %s", title)
        wikitext = self._fetch_wikitext(title)
        cache_path.write_text(wikitext, encoding="utf-8")
        time.sleep(REQUEST_DELAY)
        return wikitext

    def get_category_members(self, category: str) -> tuple[list[str], list[str]]:
        """Return (pages, subcategories) of a category, fully paginated."""
        pages: list[str] = []
        subcats: list[str] = []
        cont: str | None = None
        while True:
            params: dict = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": f"Catégorie:{category}",
                "cmtype": "page|subcat",
                "cmlimit": 500,
                "format": "json",
                "formatversion": 2,
            }
            if cont:
                params["cmcontinue"] = cont
            data = self._get_json(params)
            for member in data["query"]["categorymembers"]:
                title = member["title"]
                if member.get("type") == "subcat" or member.get("ns") == 14:
                    subcats.append(title.removeprefix("Catégorie:"))
                else:
                    pages.append(title)
            cont = data.get("continue", {}).get("cmcontinue")
            if cont is None:
                return pages, subcats

    # -- internals ------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(requests.RequestException),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
    )
    def _fetch_wikitext(self, title: str) -> str:
        data = self._get_json(
            {
                "action": "query",
                "prop": "revisions",
                "rvprop": "content",
                "rvslots": "main",
                "titles": title,
                "format": "json",
                "formatversion": 2,
            }
        )
        page = data["query"]["pages"][0]
        if "revisions" not in page:
            raise ValueError(f"Page not found: {title!r}")
        return page["revisions"][0]["slots"]["main"]["content"]

    @retry(
        retry=retry_if_exception_type(requests.RequestException),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
    )
    def _get_json(self, params: dict) -> dict:
        resp = self._session.get(API_URL, params=params, timeout=30)
        resp.raise_for_status()
        time.sleep(REQUEST_DELAY)
        return resp.json()

    def _cache_path(self, title: str) -> Path:
        digest = hashlib.sha1(title.encode()).hexdigest()[:12]
        slug = "".join(c if c.isalnum() else "_" for c in title)[:60]
        return self.cache_dir / f"{slug}_{digest}.wiki"
