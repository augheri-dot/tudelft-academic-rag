"""
Gold Layer Enrichment & Parsing Utilities
=========================================
Module providing robust fallback mechanisms for academic literature extraction:
1. OpenAlex Inverted Index Abstract Reconstruction.
2. Production-grade HTML Landing Page Content Scraping.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Optional, List
from bs4 import BeautifulSoup

logger = logging.getLogger("GoldEnrichmentUtils")


class AbstractReconstructor:
    """Reconstructs full abstract text from OpenAlex inverted index structure."""

    @staticmethod
    def reconstruct(inverted_index: Optional[Dict[str, List[int]]]) -> Optional[str]:
        """
        Transforms OpenAlex `abstract_inverted_index` dictionary back into ordered prose.
        
        Example Input: {"The": [0], "paper": [1], "presents": [2]}
        Example Output: "The paper presents"
        """
        if not inverted_index or not isinstance(inverted_index, dict):
            return None

        try:
            position_map: Dict[int, str] = {}
            for word, positions in inverted_index.items():
                for pos in positions:
                    position_map[pos] = word

            if not position_map:
                return None

            max_pos = max(position_map.keys())
            ordered_words = [position_map.get(i, "") for i in range(max_pos + 1)]
            
            # Clean up whitespace and join
            abstract_text = " ".join(filter(None, ordered_words)).strip()
            return abstract_text if len(abstract_text) > 20 else None

        except Exception as err:
            logger.debug(f"Failed to reconstruct abstract: {err}")
            return None


class HTMLLandingPageParser:
    """Extracts structured academic metadata and body content from HTML landing pages."""

    # Standard academic meta tag names used by Highwire Press, Dublin Core, and Open Graph
    ABSTRACT_META_NAMES = [
        "citation_abstract",
        "dc.description",
        "description",
        "og:description",
        "twitter:description",
    ]

    PDF_META_NAMES = [
        "citation_pdf_url",
        "dc.identifier",
    ]

    @classmethod
    def parse_html(cls, html_content: str) -> Dict[str, Optional[str]]:
        """
        Parses HTML landing pages to extract abstract, body text, or candidate PDF links.
        """
        result = {
            "extracted_abstract": None,
            "pdf_candidate_url": None,
            "html_body_text": None,
        }

        if not html_content or len(html_content.strip()) == 0:
            return result

        try:
            soup = BeautifulSoup(html_content, "html.parser")

            # 1. Extract Meta Abstract
            for meta_name in cls.ABSTRACT_META_NAMES:
                meta_tag = soup.find("meta", attrs={"name": re.compile(f"^{meta_name}$", re.I)}) or \
                           soup.find("meta", attrs={"property": re.compile(f"^{meta_name}$", re.I)})
                if meta_tag and meta_tag.get("content"):
                    result["extracted_abstract"] = meta_tag["content"].strip()
                    break

            # 2. Extract Candidate Direct PDF Link from Meta
            for meta_name in cls.PDF_META_NAMES:
                meta_tag = soup.find("meta", attrs={"name": re.compile(f"^{meta_name}$", re.I)})
                if meta_tag and meta_tag.get("content"):
                    content_val = meta_tag["content"].strip()
                    if content_val.endswith(".pdf") or "pdf" in content_val.lower():
                        result["pdf_candidate_url"] = content_val
                        break

            # 3. Fallback: Parse main HTML text content if meta abstract fails
            if not result["extracted_abstract"]:
                # Remove non-content elements
                for element in soup(["script", "style", "nav", "header", "footer", "form"]):
                    element.decompose()

                # Look for common article content tags
                main_content = soup.find("main") or soup.find("article") or soup.find("div", class_=re.compile("abstract|content|article", re.I))
                target_node = main_content if main_content else soup.body

                if target_node:
                    text_lines = [line.strip() for line in target_node.get_text(separator="\n").splitlines() if line.strip()]
                    full_text = " ".join(text_lines)
                    if len(full_text) > 100:
                        result["html_body_text"] = full_text[:10000] # Limit size to avoid excessive noise

        except Exception as err:
            logger.debug(f"HTML parsing error: {err}")

        return result
