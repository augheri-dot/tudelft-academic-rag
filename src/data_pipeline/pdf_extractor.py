import io
import requests
from typing import Optional
from pypdf import PdfReader
from tenacity import retry, stop_after_attempt, wait_exponential
from src.config.settings import settings

class PDFStreamExtractor:
    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self.headers = {
            "User-Agent": f"TUDelftAcademicRAG/1.0 (mailto:{settings.USER_EMAIL})"
        }

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5))
    def fetch_pdf_bytes(self, pdf_url: str) -> bytes:
        """Downloads PDF file directly into RAM memory bytes without saving to disk."""
        # Tuple timeout: (connection timeout, read timeout)
        response = requests.get(pdf_url, headers=self.headers, timeout=(5, self.timeout))
        response.raise_for_status()
        
        # Verify content type
        content_type = response.headers.get("Content-Type", "").lower()
        if "pdf" not in content_type and not pdf_url.lower().endswith(".pdf"):
            print(f"[WARNING] URL {pdf_url} might not be a direct PDF stream (Content-Type: {content_type})")
            
        return response.content

    def extract_full_text_from_url(self, pdf_url: Optional[str]) -> Optional[str]:
        """Streams PDF from URL into BytesIO and extracts all full-text page contents."""
        if not pdf_url:
            return None

        try:
            print(f"[INFO] Streaming PDF full-text from: {pdf_url} ...")
            pdf_bytes = self.fetch_pdf_bytes(pdf_url)
            
            # Read bytes directly from RAM
            pdf_file = io.BytesIO(pdf_bytes)
            reader = PdfReader(pdf_file)
            
            extracted_pages = []
            for idx, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    extracted_pages.append(page_text)
            
            full_text = "\n\n".join(extracted_pages).strip()
            
            if not full_text:
                print(f"[WARNING] Could not extract readable text from PDF: {pdf_url}")
                return None

            print(f"[SUCCESS] Extracted {len(reader.pages)} pages ({len(full_text)} chars) from PDF!")
            return full_text

        except Exception as e:
            print(f"[ERROR] Failed to extract full-text from PDF ({pdf_url}): {str(e)}")
            return None

if __name__ == "__main__":
    # Quick Test Execution using an open access PDF sample
    sample_pdf = "https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0280000&type=printable"
    extractor = PDFStreamExtractor()
    text = extractor.extract_full_text_from_url(sample_pdf)
    if text:
        print("\n--- SAMPLE EXTRACTED TEXT (FIRST 500 CHARS) ---")
        print(text[:500])
