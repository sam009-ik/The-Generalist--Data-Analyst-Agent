from typing import List
from bs4 import BeautifulSoup
from fastapi import UploadFile
from playwright.sync_api import sync_playwright
from html import unescape
import re
from bs4 import Tag
from playwright.sync_api import sync_playwright
import trafilatura

def render_html_url(html_urls: List[str]) -> str:
    """
    Render HTML URLs with Playwright and extract clean text using Trafilatura.
    """
    extracted_texts = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for html_url in html_urls:
            try:
                page.goto(html_url, timeout=60000)
                page.wait_for_timeout(3000)
                raw_html = page.content()
                extracted = trafilatura.extract(raw_html, include_tables=True)
                extracted_texts.append(extracted or "")
            except Exception as e:
                print(f"[Warning] Failed to render {html_url}: {e}")
                extracted_texts.append("")  # maintain order
        browser.close()

    return "\n".join(extracted_texts)

async def render_html_file(html_files: List[UploadFile]) -> str:
    """Reads uploaded HTML files and extracts clean text using Trafilatura."""
    extracted_texts = []
    for html_file in html_files:
        html_content = await html_file.read()
        decoded_html = html_content.decode("utf-8", errors="ignore")
        extracted = trafilatura.extract(decoded_html, include_tables=True)
        extracted_texts.append(extracted or "")
    return "\n".join(extracted_texts)