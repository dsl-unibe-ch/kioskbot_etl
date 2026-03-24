"""CrawlSpider that yields URLs and PDF links (not page content).

Configured entirely via a YAML file passed as ``-a config=path.yml``.
"""

import logging
from pathlib import Path
from typing import ClassVar

import scrapy
import yaml
from scrapy.http import TextResponse
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from w3lib.url import canonicalize_url

logger = logging.getLogger(__name__)


def _as_list(value: list | str | None) -> list[str]:
    """Normalise a YAML value (list, CSV string, or None) into a list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return [x.strip() for x in value.split(",") if x.strip()]


class LinkSpider(CrawlSpider):
    """Crawl a site and emit ``{"URL": ...}`` / ``{"PDF": ...}`` items.

    All configuration comes from the YAML file.
    """

    name = "link_spider"
    #NoQA: S104 Intended as a class-level constant, shared by all instances
    custom_settings: ClassVar[dict[str, object]] = {
        "ROBOTSTXT_OBEY": True,
        "LOG_LEVEL": "INFO",
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 0.5,
        "AUTOTHROTTLE_MAX_DELAY": 5.0,
        "DOWNLOAD_DELAY": 0.25,
        "DEPTH_LIMIT": 12,
    }

    def __init__(self, config: str, **kwargs: str) -> None:
        super().__init__(**kwargs)

        with Path(config).open(encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f) or {}

        self.static_pdfs: list[str] = yaml_config.get("static_pdfs", [])
        self.start_urls = _as_list(yaml_config.get("seed_urls"))
        self.allowed_domains = _as_list(yaml_config.get("allowed_domains"))

        link_extractor = LinkExtractor(
            allow=tuple(_as_list(yaml_config.get("allow"))),
            deny=tuple(_as_list(yaml_config.get("deny_domains"))),
            allow_domains=self.allowed_domains or (),
            deny_domains=tuple(_as_list(yaml_config.get("deny_domains"))),
            unique=True,
        )
        self.rules = (Rule(link_extractor, callback="parse_item", follow=True),)
        self._compile_rules()

        if not self.start_urls:
            self.logger.warning("No seed_urls in config %s", config)

        self._exported: set[str] = set()

    def _canon(self, url: str) -> str:
        return canonicalize_url(url, keep_fragments=False)

    def start_requests(self):
        for pdf_url in self.static_pdfs:
            canonical = self._canon(pdf_url)
            if canonical not in self._exported:
                self._exported.add(canonical)
                yield {"PDF": canonical}

        for url in self.start_urls:
            yield scrapy.Request(url, dont_filter=True)

    def parse_start_url(self, response: scrapy.http.Response):
        return self.parse_item(response)

    def parse_item(self, response: scrapy.http.Response):
        url = self._canon(response.url)
        if url not in self._exported:
            self._exported.add(url)
            yield {"URL": url}

        if not isinstance(response, TextResponse):
            return

        for href in response.css("a::attr(href)").getall():
            if ".pdf" in href.lower():
                pdf_url = self._canon(response.urljoin(href))
                if pdf_url not in self._exported:
                    self._exported.add(pdf_url)
                    yield {"PDF": pdf_url}
