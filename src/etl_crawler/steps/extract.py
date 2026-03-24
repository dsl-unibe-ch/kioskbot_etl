"""Step 2 — Extract: pull content from URLs and PDFs into content.jsonl."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin, urlparse

import pdfplumber
import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from playwright.async_api import (
    Page,
    TimeoutError as PlaywrightTimeout,
    async_playwright,
)

#NoQA: F401 Used to prevent circular import of RunContext from pipeline.py during static type checking
if TYPE_CHECKING:
    from src.etl_crawler.pipeline import RunContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# URL content extraction (Playwright + BeautifulSoup)
# ---------------------------------------------------------------------------


class ContentExtractor:
    """Expand accordions/tabs then extract text in reading order."""

    def __init__(self, customer_name: str, output_dir: Path) -> None:
        self.customer_name = customer_name
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.current_base_url: str | None = None

    async def expand_all_accordions(self, page: Page) -> int:
        """Expand all accordions on the page."""
        expanded_count = 0
        accordion_selectors = [
            "details:not([open])",
            '[aria-expanded="false"]',
            '[class*="accordion"]:not(.expanded)',
            '[class*="collapse"]:not(.show)',
            'button[class*="accordion"]',
        ]
        for selector in accordion_selectors:
            try:
                elements = await page.query_selector_all(selector)
                for element in elements:
                    try:
                        if await element.is_visible():
                            tag_name = await element.evaluate("el => el.tagName.toLowerCase()")
                            if tag_name == "details":
                                await element.evaluate('el => el.setAttribute("open", "")')
                            else:
                                await element.click(timeout=1000)
                            expanded_count += 1
                            await asyncio.sleep(0.1)
                    except Exception:
                        pass
            except Exception:
                pass
        return expanded_count

    async def click_all_tabs(self, page: Page) -> int:
        """Click all tabs on the page."""
        tab_selectors = [
            '[role="tab"]',
            '[data-toggle="tab"]',
            'button[class*="tab"]:not([class*="table"])',
            'a[class*="tab"]:not([class*="table"])',
        ]
        tabs: list = []
        for selector in tab_selectors:
            try:
                found = await page.query_selector_all(selector)
                for tab in found:
                    if await tab.is_visible():
                        tabs.append(tab)
                if tabs:
                    break
            except Exception:
                pass

        for i, tab in enumerate(tabs):
            try:
                await tab.click(timeout=2000)
                await asyncio.sleep(0.3)
                await self.expand_all_accordions(page)
            except Exception:
                pass

        if tabs:
            try:
                await tabs[0].click(timeout=2000)
                await asyncio.sleep(0.3)
            except Exception:
                pass
        return len(tabs)

    def extract_text_content(self, soup: BeautifulSoup) -> str:
        """Extract text content from the page."""
        main_content = None
        for selector in ["main", '[role="main"]', "article", ".content", "#content", "body"]:
            main_content = soup.select_one(selector)
            if main_content:
                break
        if not main_content:
            main_content = soup.body if soup.body else soup

        for element in main_content.find_all(["script", "style", "noscript"]):
            element.decompose()
        for element in main_content.find_all(style=lambda s: s and "display:none" in s.replace(" ", "")):
            element.decompose()
        for element in main_content.find_all(attrs={"hidden": True}):
            element.decompose()
        for element in main_content.find_all(attrs={"aria-hidden": "true"}):
            element.decompose()

        lines: list[str] = []
        self._extract_text_recursive(main_content, lines, level=0)
        result = "\n".join(lines)
        while "\n\n\n" in result:
            result = result.replace("\n\n\n", "\n\n")
        return result.strip()

    def _extract_text_recursive(self, element: Any, lines: list[str], level: int = 0) -> None:
        """Extract text content from the element recursively."""
        if isinstance(element, NavigableString):
            text = str(element).strip()
            if text:
                lines.append(text)
            return
        if not isinstance(element, Tag):
            return
        if element.name in ("script", "style", "noscript"):
            return

        if element.name == "a":
            text = element.get_text(strip=True)
            href = element.get("href", "")
            if text and href:
                absolute_url = urljoin(self.current_base_url, href) if self.current_base_url else href
                if absolute_url.startswith("http"):
                    lines.append(f"{text} {absolute_url}")
                else:
                    lines.append(text)
            elif text:
                lines.append(text)
            return

        if element.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            text = element.get_text(strip=True)
            if text:
                if lines and lines[-1]:
                    lines.append("")
                lines.append(text + " ")
                lines.append("")
        elif element.name in ("p", "div", "section", "article"):
            for child in element.children:
                self._extract_text_recursive(child, lines, level + 1)
            if element.name in ("p", "section") and lines and lines[-1]:
                lines.append("")
        elif element.name == "li":
            for child in element.children:
                self._extract_text_recursive(child, lines, level + 1)
        elif element.name == "br":
            lines.append("")
        elif element.name == "summary":
            text = element.get_text(strip=True)
            if text:
                if lines and lines[-1]:
                    lines.append("")
                lines.append(text + " ")
        else:
            for child in element.children:
                self._extract_text_recursive(child, lines, level)

    async def process_url(self, page: Page, url: str, index: int) -> dict[str, Any]:
        """Process the URL and extract text content."""
        self.current_base_url = url
        try:
            await page.goto(url, wait_until="networkidle", timeout=30000)
            await asyncio.sleep(1)
            await self.expand_all_accordions(page)
            await self.click_all_tabs(page)
            html = await page.content()
            soup = BeautifulSoup(html, "html.parser")
            content = self.extract_text_content(soup)
            return {
                "url": url,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "content": content,
                "success": True,
            }
        except PlaywrightTimeout as e:
            logger.error("Timeout processing %s: %s", url, e)
            return {"url": url, "timestamp": datetime.now(timezone.utc).isoformat(), "error": str(e), "success": False}
        except Exception as e:
            logger.error("Error processing %s: %s", url, e)
            return {"url": url, "timestamp": datetime.now(timezone.utc).isoformat(), "error": str(e), "success": False}

    def save_result(self, result: dict[str, Any], output_file: Path) -> None:
        """Save the result to the output file."""
        with output_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    async def process_urls(self, urls: list[str], output_file: Path) -> None:
        """Process the URLs and extract text content."""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            )
            page = await context.new_page()
            try:
                for i, url in enumerate(urls, start=1):
                    result = await self.process_url(page, url, i)
                    self.save_result(result, output_file)
                    await asyncio.sleep(1)
            finally:
                await browser.close()


# ---------------------------------------------------------------------------
# PDF content extraction
# ---------------------------------------------------------------------------


class PDFContentExtractor:
    """Download and extract text from PDFs."""

    def __init__(self, customer_name: str, output_dir: Path, download_dir: Path | None = None) -> None:
        self.customer_name = customer_name
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.download_dir = download_dir or (output_dir / "raw" / "pdf_files")
        self.download_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def is_url(path_str: str) -> bool:
        """Check if the path is a URL."""
        return path_str.startswith("http://") or path_str.startswith("https://")

    def download_pdf(self, url: str) -> Path | None:
        """Download the PDF from the URL."""
        try:
            parsed_url = urlparse(url)
            filename = Path(parsed_url.path).name
            if not filename or not filename.endswith(".pdf"):
                import hashlib
                url_hash = hashlib.md5(url.encode()).hexdigest()[:8]  # noqa: S324
                filename = f"downloaded_{url_hash}.pdf"
            local_path = self.download_dir / filename
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()
            with local_path.open("wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return local_path
        except Exception as e:
            logger.error("Error downloading PDF from %s: %s", url, e)
            return None

    @staticmethod
    def extract_text_from_pdf(pdf_path: Path) -> str:
        """Extract text content from the PDF."""
        content_parts: list[str] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                try:
                    text = page.extract_text()
                    if text:
                        content_parts.append(text)
                except Exception:
                    pass
        return "\n\n".join(content_parts)

    def process_pdf(self, pdf_path: Path, index: int, original_url: str | None = None) -> dict[str, Any]:
        """Process the PDF and extract text content."""
        try:
            content = self.extract_text_from_pdf(pdf_path)
            while "\n\n\n" in content:
                content = content.replace("\n\n\n", "\n\n")
            content = content.strip()
            return {
                "url": original_url or str(pdf_path.absolute()),
                "filename": pdf_path.name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "content": content,
                "success": True,
            }
        except Exception as e:
            logger.error("Error processing %s: %s", pdf_path.name, e)
            return {
                "url": original_url or str(pdf_path.absolute()),
                "filename": pdf_path.name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
                "success": False,
            }

    def process_pdfs(self, pdf_links: list[str], output_file: Path) -> None:
        """Process the PDFs and extract text content."""
        for i, pdf_link in enumerate(pdf_links, start=1):
            downloaded_file = None
            original_url = None
            try:
                if self.is_url(pdf_link):
                    original_url = pdf_link
                    downloaded_file = self.download_pdf(pdf_link)
                    if not downloaded_file:
                        continue
                    pdf_path = downloaded_file
                else:
                    pdf_path = Path(pdf_link)
                    if not pdf_path.exists():
                        logger.error("PDF file not found: %s", pdf_path)
                        continue

                result = self.process_pdf(pdf_path, i, original_url)
                if result.get("success"):
                    with output_file.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
            finally:
                if downloaded_file and downloaded_file.exists():
                    try:
                        downloaded_file.unlink()
                    except Exception:
                        pass


# ---------------------------------------------------------------------------
# Step entry-point
# ---------------------------------------------------------------------------


@dataclass
class ExtractResult:
    content_jsonl_path: Path
    url_count: int
    pdf_count: int


def run(run_context: RunContext) -> ExtractResult:
    """Run URL + PDF extraction for all items in url_list.jsonl."""
    url_list_path = run_context.data_dir / "url_list.jsonl"
    if not url_list_path.exists():
        raise FileNotFoundError(f"url_list.jsonl not found in {run_context.data_dir}")

    urls: list[str] = []
    pdf_links: list[str] = []
    seen_urls: set[str] = set()
    seen_pdfs: set[str] = set()
    with url_list_path.open("r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            if "PDF" in data:
                pdf_url = data["PDF"]
                if pdf_url not in seen_pdfs:
                    seen_pdfs.add(pdf_url)
                    pdf_links.append(pdf_url)
                continue

            if "URL" in data:
                url = data["URL"]
                if url.lower().endswith(".pdf"):
                    if url not in seen_pdfs:
                        seen_pdfs.add(url)
                        pdf_links.append(url)
                elif url not in seen_urls:
                    seen_urls.add(url)
                    urls.append(url)

    output_file = run_context.data_dir / f"{run_context.customer_name}_content.jsonl"
    logger.info("Extracting content from %d URLs and %d PDFs", len(urls), len(pdf_links))

    # Scrapy/Twisted may have corrupted the event loop, so force a fresh one
    # before spawning the Playwright browser subprocess.
    if urls:
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            extractor = ContentExtractor(run_context.customer_name, run_context.data_dir)
            loop.run_until_complete(extractor.process_urls(urls, output_file))
        finally:
            loop.close()

    if pdf_links:
        pdf_extractor = PDFContentExtractor(
            run_context.customer_name,
            run_context.data_dir,
            download_dir=run_context.data_dir / "raw" / "pdf_files",
        )
        pdf_extractor.process_pdfs(pdf_links, output_file)

    result = ExtractResult(
        content_jsonl_path=output_file,
        url_count=len(urls),
        pdf_count=len(pdf_links),
    )
    logger.info("Extraction finished: %d URLs, %d PDFs -> %s",
                result.url_count, result.pdf_count, result.content_jsonl_path)
    return result
