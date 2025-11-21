"""Web documentation loader."""

from typing import Any, Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse
import time

import requests
from bs4 import BeautifulSoup
import html2text

from src.utils.interfaces import DocumentLoader


class WebDocumentationLoader(DocumentLoader):
    """Load documentation from websites with configurable extraction."""

    def __init__(
        self,
        preserve_structure: bool = False,
        extract_code_blocks: bool = True,
        extract_headers: bool = True,
        timeout: int = 30,
        user_agent: str = "RAG-Demo-Bot/1.0",
        delay_between_requests: float = 0.5,
    ):
        """
        Initialize web documentation loader.

        Args:
            preserve_structure: Whether to preserve HTML structure
            extract_code_blocks: Whether to extract code blocks
            extract_headers: Whether to extract headers
            timeout: Request timeout in seconds
            user_agent: User agent string
            delay_between_requests: Delay between requests (rate limiting)
        """
        self.preserve_structure = preserve_structure
        self.extract_code_blocks = extract_code_blocks
        self.extract_headers = extract_headers
        self.timeout = timeout
        self.user_agent = user_agent
        self.delay_between_requests = delay_between_requests
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = True

    def load(self, source: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Load a single web page.

        Args:
            source: URL to load
            **kwargs: Additional parameters

        Returns:
            List containing single document with content and metadata
        """
        try:
            response = requests.get(
                source,
                timeout=self.timeout,
                headers={"User-Agent": self.user_agent},
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # Extract main content
            content = self._extract_content(soup)

            # Extract metadata
            metadata = self._extract_metadata(soup, source)

            return [{"text": content, "metadata": metadata}]

        except Exception as e:
            print(f"Error loading {source}: {e}")
            return []

    def load_batch(
        self,
        sources: List[str],
        max_depth: int = 1,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Load documents from multiple sources with crawling.

        Args:
            sources: List of starting URLs
            max_depth: Maximum crawl depth
            include_patterns: URL patterns to include
            exclude_patterns: URL patterns to exclude

        Returns:
            List of documents with content and metadata
        """
        visited: Set[str] = set()
        to_visit: List[tuple[str, int]] = [(url, 0) for url in sources]
        documents = []

        while to_visit:
            url, depth = to_visit.pop(0)

            # Skip if already visited or max depth reached
            if url in visited or depth > max_depth:
                continue

            # Check patterns
            if not self._should_include_url(url, include_patterns, exclude_patterns):
                continue

            visited.add(url)
            print(f"Loading: {url} (depth {depth})")

            # Load document
            docs = self.load(url)
            if docs:
                documents.extend(docs)

                # Extract links for further crawling
                if depth < max_depth:
                    links = self._extract_links(url, docs[0].get("soup"))
                    for link in links:
                        if link not in visited:
                            to_visit.append((link, depth + 1))

            # Rate limiting
            time.sleep(self.delay_between_requests)

        return documents

    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract content from HTML."""
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()

        # Try to find main content area
        main_content = (
            soup.find("main")
            or soup.find("article")
            or soup.find("div", class_=["content", "documentation", "main", "body"])
            or soup.find("body")
        )

        if not main_content:
            main_content = soup

        if self.preserve_structure:
            # Convert HTML to markdown preserving structure
            return self.html_converter.handle(str(main_content))
        else:
            # Extract clean text
            text = main_content.get_text(separator="\n", strip=True)

            # Clean up excessive whitespace
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            return "\n".join(lines)

    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract metadata from HTML."""
        metadata = {"source": url, "url": url}

        # Title
        title_tag = soup.find("title")
        if title_tag:
            metadata["title"] = title_tag.get_text().strip()

        # Meta description
        desc_tag = soup.find("meta", attrs={"name": "description"})
        if desc_tag and desc_tag.get("content"):
            metadata["description"] = desc_tag["content"].strip()

        # Headers
        if self.extract_headers:
            headers = []
            for i in range(1, 4):  # h1, h2, h3
                for header in soup.find_all(f"h{i}"):
                    headers.append(header.get_text().strip())
            if headers:
                metadata["headers"] = headers[:10]  # Limit to first 10

        # Code blocks
        if self.extract_code_blocks:
            code_blocks = []
            for code in soup.find_all("code"):
                code_text = code.get_text().strip()
                if len(code_text) > 10:  # Only meaningful code blocks
                    code_blocks.append(code_text)
            if code_blocks:
                metadata["code_blocks_count"] = len(code_blocks)

        return metadata

    def _extract_links(self, base_url: str, soup: Optional[BeautifulSoup]) -> List[str]:
        """Extract links from HTML for crawling."""
        if not soup:
            return []

        links = []
        base_domain = urlparse(base_url).netloc

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            full_url = urljoin(base_url, href)

            # Only include links from same domain
            if urlparse(full_url).netloc == base_domain:
                # Remove fragments
                full_url = full_url.split("#")[0]
                links.append(full_url)

        return list(set(links))

    def _should_include_url(
        self,
        url: str,
        include_patterns: Optional[List[str]],
        exclude_patterns: Optional[List[str]],
    ) -> bool:
        """Check if URL should be included based on patterns."""
        # Check exclude patterns first
        if exclude_patterns:
            for pattern in exclude_patterns:
                if pattern in url:
                    return False

        # Check include patterns
        if include_patterns:
            for pattern in include_patterns:
                if pattern in url:
                    return True
            return False  # If include patterns specified but none matched

        return True  # No patterns specified, include all


class DocumentLoaderFactory:
    """Factory for creating document loaders."""

    @staticmethod
    def create(loader_type: str, config: Dict[str, Any]) -> DocumentLoader:
        """
        Create a document loader.

        Args:
            loader_type: Loader type (web, file, api)
            config: Loader configuration

        Returns:
            DocumentLoader instance

        Raises:
            ValueError: If loader type is unknown
        """
        if loader_type == "web":
            return WebDocumentationLoader(
                preserve_structure=config.get("preserve_structure", False),
                extract_code_blocks=config.get("extract_code_blocks", True),
                extract_headers=config.get("extract_headers", True),
                timeout=config.get("timeout", 30),
                user_agent=config.get("user_agent", "RAG-Demo-Bot/1.0"),
                delay_between_requests=config.get("delay_between_requests", 0.5),
            )

        else:
            raise ValueError(
                f"Unknown document loader type: {loader_type}. " f"Supported: web"
            )
