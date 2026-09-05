import io
import requests
import fitz
from typing import Optional, Dict, Any, List


class OpenAlexPDFExtractor:
    """Production-ready module for downloading PDFs to RAM streams and extracting text with PyMuPDF."""

    DEFAULT_USER_AGENT = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    def __init__(self, user_agent: Optional[str] = None):
        self.headers = {
            "User-Agent": user_agent or self.DEFAULT_USER_AGENT,
            "Accept": "application/pdf,*/*"
        }

    def download_pdf_to_ram(self, pdf_url: str, timeout: int = 10) -> Optional[io.BytesIO]:
        """Downloads PDF payload directly into a RAM stream with byte header validation."""
        try:
            response = requests.get(
                pdf_url, 
                headers=self.headers, 
                timeout=(4, timeout), 
                stream=True, 
                allow_redirects=True
            )
            
            if response.status_code != 200:
                print(f"[WARN] HTTP {response.status_code} on URL: {pdf_url}")
                return None

            content = response.content
            
            if b"%PDF" not in content[:1024]:
                print(f"[WARN] Payload from {pdf_url} is not a valid PDF binary (magic bytes missing).")
                return None

            return io.BytesIO(content)

        except Exception as e:
            print(f"[ERROR] Failed to download PDF stream from {pdf_url}: {e}")
            return None

    def extract_text_from_pdf_stream(self, pdf_stream: io.BytesIO) -> Dict[str, Any]:
        """Extracts text page-by-page from an in-memory PDF stream via PyMuPDF."""
        extracted_data = {
            "full_text": "",
            "pages_count": 0,
            "char_count": 0
        }
        
        try:
            with fitz.open(stream=pdf_stream, filetype="pdf") as doc:
                extracted_data["pages_count"] = len(doc)
                full_text_list = []
                
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text = page.get_text("text")
                    if text.strip():
                        full_text_list.append(text)
                
                full_text = "\n\n".join(full_text_list)
                extracted_data["full_text"] = full_text
                extracted_data["char_count"] = len(full_text)

        except Exception as e:
            print(f"[ERROR] PyMuPDF extraction failed: {e}")

        return extracted_data

    def _extract_candidate_urls(self, pub_data: Dict[str, Any]) -> List[str]:
        """Collects and prioritizes candidate PDF URLs from OpenAlex record payload."""
        urls = []
        locations = []

        if pub_data.get("best_oa_location"):
            locations.append(pub_data["best_oa_location"])
        if pub_data.get("primary_location"):
            locations.append(pub_data["primary_location"])
        locations.extend(pub_data.get("locations") or [])

        for loc in locations:
            url = loc.get("pdf_url")
            if url and isinstance(url, str):
                if url.lower().endswith(".pdf") or "repository.tudelft.nl" in url or "arxiv.org" in url:
                    if url not in urls:
                        urls.append(url)

        for loc in locations:
            url = loc.get("pdf_url")
            if url and isinstance(url, str) and url not in urls:
                urls.append(url)

        root_url = pub_data.get("pdf_url")
        if root_url and isinstance(root_url, str) and root_url not in urls:
            urls.append(root_url)

        return urls

    def process_openalex_publication(self, pub_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main orchestrator for attempting full-text extraction across candidate links."""
        candidate_urls = self._extract_candidate_urls(pub_data)

        if not candidate_urls:
            print(f"[INFO] No Open Access PDF URLs found for: {pub_data.get('title', 'Untitled')}")
            pub_data["pdf_extraction_status"] = "NO_URL"
            return pub_data

        for idx, pdf_url in enumerate(candidate_urls, 1):
            print(f"[INFO] Attempt {idx}/{len(candidate_urls)} for '{pub_data.get('title', 'Untitled')}'")
            print(f"[INFO] Downloading: {pdf_url}")
            
            pdf_stream = self.download_pdf_to_ram(pdf_url)
            if not pdf_stream:
                continue

            extraction_result = self.extract_text_from_pdf_stream(pdf_stream)
            
            if extraction_result["char_count"] >= 200:
                pub_data["full_text"] = extraction_result["full_text"]
                pub_data["pdf_extraction_status"] = "SUCCESS"
                pub_data["pdf_pages_count"] = extraction_result["pages_count"]
                pub_data["pdf_char_count"] = extraction_result["char_count"]
                print(f"[SUCCESS] Extracted {extraction_result['char_count']} chars across {extraction_result['pages_count']} pages.\n")
                return pub_data
            else:
                print(f"[WARN] Extracted text too short ({extraction_result['char_count']} chars). Trying next candidate...\n")

        pub_data["pdf_extraction_status"] = "FAILED_ALL_CANDIDATES"
        return pub_data
