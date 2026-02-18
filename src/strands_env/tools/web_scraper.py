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
and optionally uses a Bedrock LLM to extract task-relevant information.

Content extraction pipeline:
  1. trafilatura: extracts main content, strips boilerplate (primary)
  2. html2text: full HTML-to-Markdown conversion (fallback)

Example:
    >>> from strands_env.tools import WebScraperToolkit
    >>> from strands_env.utils.aws import get_session
    >>> toolkit = WebScraperToolkit(boto3_session=get_session(region="us-west-2"))
    >>> result = toolkit.scrape("https://example.com", instruction="What is the main topic?")
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import html2text
import requests
import trafilatura
from strands import tool

if TYPE_CHECKING:
    import boto3

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30
MAX_CONTENT_CHARS = 20000
MAX_OUTPUT_TOKENS = 2048
DEFAULT_MODEL_ID = "amazon.nova-lite-v1:0"

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
    """Web scraper with Bedrock LLM extraction for strands agents.

    Fetches web pages and extracts relevant content using trafilatura for
    main content extraction and a Bedrock LLM for targeted extraction
    based on a user-provided instruction.

    Example:
        from strands_env.tools import WebScraperToolkit
        from strands_env.utils.aws import get_session

        toolkit = WebScraperToolkit(boto3_session=get_session(region="us-west-2"))

        class MyEnv(Environment):
            def __init__(self, boto3_session, ...):
                self.toolkit = WebScraperToolkit(boto3_session=boto3_session)

            def get_tools(self):
                return [self.toolkit.scrape]
    """

    def __init__(
        self,
        boto3_session: boto3.Session,
        model_id: str = DEFAULT_MODEL_ID,
        max_content_chars: int = MAX_CONTENT_CHARS,
        max_output_tokens: int = MAX_OUTPUT_TOKENS,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        """Initialize Web Scraper Toolkit.

        Args:
            boto3_session: boto3 session for Bedrock credentials.
            model_id: Bedrock model ID for LLM extraction.
            max_content_chars: Max characters of page content to send to LLM.
            max_output_tokens: Max tokens for LLM extraction output.
            timeout: HTTP request timeout in seconds.
        """
        self._bedrock_client = boto3_session.client("bedrock-runtime")
        self.model_id = model_id
        self.max_content_chars = max_content_chars
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout

        self._h2t = html2text.HTML2Text()
        self._h2t.ignore_links = False
        self._h2t.ignore_images = True
        self._h2t.ignore_emphasis = False
        self._h2t.body_width = 0

        self._scrape = self._create_scrape_tool()

    def _fetch_html(self, url: str) -> str:
        """Fetch HTML content from URL."""
        response = requests.get(url, timeout=self.timeout, headers=_REQUEST_HEADERS)
        response.raise_for_status()
        return response.text

    def _extract_main_content(self, html: str, url: str) -> str:
        """Extract main content from HTML, stripping boilerplate.

        Uses trafilatura as primary extractor; falls back to html2text
        for pages where trafilatura returns insufficient content.
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
        return self._h2t.handle(html)

    def _truncate(self, text: str) -> str:
        """Truncate text to max_content_chars."""
        if len(text) > self.max_content_chars:
            return text[: self.max_content_chars] + "\n\n[Content truncated...]"
        return text

    def _extract_with_llm(self, content: str, instruction: str) -> str:
        """Use Bedrock LLM to extract relevant information from content."""
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(
            instruction=instruction,
            content=self._truncate(content),
        )
        response = self._bedrock_client.converse(
            modelId=self.model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": self.max_output_tokens, "temperature": 0.0},
        )
        return response["output"]["message"]["content"][0]["text"]

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
            a Bedrock LLM to extract only the information relevant to the
            instruction.

            Args:
                url: The URL of the web page to scrape.
                instruction: What information to extract. If provided, a Bedrock
                    LLM will extract only relevant content. If empty, returns
                    the raw text (truncated to 20,000 characters).

            Returns:
                JSON with the scraped content or an error message.
            """
            import asyncio

            logger.info(f"[scrape] url={url}, instruction={instruction[:100] if instruction else '(none)'}")

            try:
                html = await asyncio.to_thread(fetch_html, url)
                main_content = await asyncio.to_thread(extract_main_content, html, url)

                if instruction:
                    content = await asyncio.to_thread(extract_with_llm, main_content, instruction)
                else:
                    content = truncate(main_content)

                return json.dumps({"ScrapedContent": {"url": url, "content": content}})

            except requests.exceptions.Timeout:
                logger.error(f"[scrape] timeout for url: {url}")
                return json.dumps({"ScrapedContent": {"url": url, "Error": "Request timeout"}})
            except requests.exceptions.HTTPError as e:
                logger.error(f"[scrape] HTTP error for url {url}: {e}")
                return json.dumps({"ScrapedContent": {"url": url, "Error": f"HTTP {e.response.status_code}"}})
            except Exception as e:
                logger.error(f"[scrape] error for url {url}: {e}")
                return json.dumps({"ScrapedContent": {"url": url, "Error": str(e)}})

        return scrape

    @property
    def scrape(self):
        """Get the scrape tool."""
        return self._scrape
