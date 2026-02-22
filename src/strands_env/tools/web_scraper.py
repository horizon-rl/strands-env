# Copyright 2025 Horizon RL Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Web scraper toolkit with LLM-based content extraction.

Fetches a web page, extracts main content (stripping nav/sidebar/ads),
and optionally uses a strands Agent to extract task-relevant information.

Content extraction pipeline:
  1. trafilatura: extracts main content, strips boilerplate (primary)
  2. html2text: full HTML-to-Markdown conversion (fallback)

Example:
    >>> from strands.models.bedrock import BedrockModel
    >>> from strands_env.tools import WebScraperToolkit
    >>> model = BedrockModel(model_id="amazon.nova-lite-v1:0")
    >>> toolkit = WebScraperToolkit(model=model)
    >>> result = toolkit.scrape("https://example.com", instruction="What is the main topic?")
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

import aiohttp
import html2text
import trafilatura
from strands import Agent, tool

if TYPE_CHECKING:
    from strands.models import Model

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30
MAX_CONTENT_CHARS = 20000

EXTRACTION_PROMPT_TEMPLATE = """Extract information relevant to the following instruction from the web page content below.
Be concise and focus on facts, data, and key details. Omit navigation, ads, and irrelevant content.

## Instruction
{instruction}

## Web Page Content
{content}"""

_REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


class WebScraperToolkit:
    """Web scraper with LLM extraction for strands agents.

    Fetches web pages and extracts relevant content using trafilatura for
    main content extraction and a strands Agent for targeted extraction
    based on a user-provided instruction.

    Example:
        from strands.models.bedrock import BedrockModel
        from strands_env.tools import WebScraperToolkit

        model = BedrockModel(model_id="amazon.nova-lite-v1:0")
        toolkit = WebScraperToolkit(model=model)

        class MyEnv(Environment):
            def __init__(self, ...):
                self.toolkit = WebScraperToolkit(model=model)

            def get_tools(self):
                return [self.toolkit.scrape]
    """

    def __init__(
        self,
        model: Model,
        max_content_chars: int = MAX_CONTENT_CHARS,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        """Initialize Web Scraper Toolkit.

        Args:
            model: A strands Model instance for LLM extraction (e.g. BedrockModel, OpenAIModel).
            max_content_chars: Max characters of page content to send to LLM.
            timeout: HTTP request timeout in seconds.
        """
        self.model = model
        self.max_content_chars = max_content_chars
        self.timeout = timeout

        self._scrape = self._create_scrape_tool()

    async def _fetch_html(self, url: str) -> str:
        """Fetch HTML content from URL."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                headers=_REQUEST_HEADERS,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                response.raise_for_status()
                return await response.text()

    @staticmethod
    def _extract_main_content(html: str, url: str) -> str:
        """Extract main content from HTML, stripping boilerplate.

        Uses trafilatura as primary extractor; falls back to html2text
        for pages where trafilatura returns insufficient content.

        A fresh html2text instance is created per call for thread safety
        (this method runs in a thread pool via asyncio.to_thread).
        """
        content = trafilatura.extract(
            html,
            url=url,
            include_links=True,
            include_tables=True,
            output_format="txt",
        )
        if content and len(content.strip()) > 100:
            return content
        h2t = html2text.HTML2Text()
        h2t.ignore_links = False
        h2t.ignore_images = True
        h2t.ignore_emphasis = False
        h2t.body_width = 0
        return h2t.handle(html)

    def _truncate(self, text: str) -> str:
        """Truncate text to max_content_chars."""
        if len(text) > self.max_content_chars:
            return text[: self.max_content_chars] + "\n\n[Content truncated...]"
        return text

    async def _extract_with_llm(self, content: str, instruction: str) -> str:
        """Use a strands Agent to extract relevant information from content."""
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(
            instruction=instruction,
            content=self._truncate(content),
        )
        agent = Agent(model=self.model, callback_handler=None)
        result = await agent.invoke_async(prompt)
        return result.message["content"][0]["text"]

    def _create_scrape_tool(self):
        """Create scrape tool."""
        fetch_html = self._fetch_html
        extract_main_content = self._extract_main_content
        extract_with_llm = self._extract_with_llm
        truncate = self._truncate

        @tool
        async def scrape(url: str, instruction: str = "") -> str:
            """Fetch a web page and extract relevant information.

            Retrieves the full HTML, converts to text, then optionally uses
            an LLM agent to extract only the information relevant to the
            instruction.

            Args:
                url: The URL of the web page to scrape.
                instruction: What information to extract. If provided, an LLM
                    agent will extract only relevant content. If empty, returns
                    the raw text (truncated).

            Returns:
                JSON with the scraped content or an error message.
            """
            logger.info(f"[scrape] url={url}, instruction={instruction[:100] if instruction else '(none)'}")

            try:
                html = await fetch_html(url)
                main_content = await asyncio.to_thread(extract_main_content, html, url)

                if instruction:
                    content = await extract_with_llm(main_content, instruction)
                else:
                    content = truncate(main_content)

                return json.dumps({"ScrapedContent": {"url": url, "content": content}})

            except TimeoutError:
                logger.error(f"[scrape] timeout for url: {url}")
                return json.dumps({"ScrapedContent": {"url": url, "Error": "Request timeout"}})
            except aiohttp.ClientResponseError as e:
                logger.error(f"[scrape] HTTP error for url {url}: {e}")
                return json.dumps({"ScrapedContent": {"url": url, "Error": f"HTTP {e.status}"}})
            except Exception as e:
                logger.error(f"[scrape] error for url {url}: {e}")
                return json.dumps({"ScrapedContent": {"url": url, "Error": str(e)}})

        return scrape

    @property
    def scrape(self):
        """Get the scrape tool."""
        return self._scrape
