import os
import json
import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Any

SILVER_PATH = "/app/data/silver/cleaned_publications.json"

class OpenAlexIngester:
    def __init__(self, email: str = "adminlibrary@tudelft.nl"):
        self.base_url = "https://api.openalex.org/works"
        self.headers = {"User-Agent": f"TUDelftAcademicRAG/1.0 (mailto:{email})"}

    def fetch_tudelft_publications(self, limit: int = 20, page: int = 1) -> List[Dict[str, Any]]:
        """Fetch publications associated with TU Delft (I865915315) with pagination support."""
        params = {
            "filter": "institutions.id:I865915315",
            "sort": "publication_date:desc",
            "per_page": min(limit, 200),
            "page": page
        }
        print(f"[INFO] Fetching page {page} ({limit} items) from OpenAlex API...")
        try:
            response = requests.get(self.base_url, params=params, headers=self.headers, timeout=15)
            if response.status_code != 200:
                print(f"[ERROR] OpenAlex API request failed with status {response.status_code}")
                return []

            data = response.json()
            results = data.get("results", [])
            print(f"[SUCCESS] Retrieved {len(results)} records from OpenAlex API (Page {page}).")
            
            cleaned_records = []
            for item in results:
                authorships = item.get("authorships", [])
                authors = [a.get("author", {}).get("display_name") for a in authorships if a.get("author")]
                
                abstract_inv = item.get("abstract_inverted_index")
                abstract_text = ""
                if abstract_inv:
                    word_positions = []
                    for word, positions in abstract_inv.items():
                        for pos in positions:
                            word_positions.append((pos, word))
                    word_positions.sort()
                    abstract_text = " ".join([w for _, w in word_positions])

                record = {
                    "openalex_id": item.get("id"),
                    "doi": item.get("doi"),
                    "title": item.get("title"),
                    "publication_year": item.get("publication_year"),
                    "authors": authors,
                    "cited_by_count": item.get("cited_by_count", 0),
                    "is_open_access": item.get("open_access", {}).get("is_oa", False),
                    "pdf_url": item.get("open_access", {}).get("oa_url"),
                    "abstract": abstract_text,
                    "executive_summary": None,
                    "expertise_metadata": None
                }
                cleaned_records.append(record)
                
            return cleaned_records
        except Exception as e:
            print(f"[ERROR] Ingestion exception on page {page}: {str(e)}")
            return []

    def save_append_silver(self, new_records: List[Dict[str, Any]]):
        """Appends new publications to existing Silver JSON layer safely without data loss."""
        if not new_records:
            print("[WARNING] No new records provided for appending.")
            return

        os.makedirs(os.path.dirname(SILVER_PATH), exist_ok=True)
        
        existing_records = []
        if os.path.exists(SILVER_PATH):
            try:
                with open(SILVER_PATH, 'r', encoding='utf-8') as f:
                    existing_records = json.load(f)
                print(f"[INFO] Existing Silver layer found with {len(existing_records)} records.")
            except Exception as e:
                print(f"[WARNING] Could not read existing Silver layer ({str(e)}). Starting fresh.")

        combined_records = existing_records + new_records
        
        seen_titles = set()
        deduped_records = []
        for rec in combined_records:
            t = (rec.get("title") or rec.get("openalex_id") or "").strip().lower()
            if t and t not in seen_titles:
                seen_titles.add(t)
                deduped_records.append(rec)

        print(f"[INFO] Deduplicated total records: {len(combined_records)} -> {len(deduped_records)} unique publications.")

        with open(SILVER_PATH, 'w', encoding='utf-8') as f:
            json.dump(deduped_records, f, indent=2, ensure_ascii=False)
            
        print(f"[SUCCESS] Silver layer updated safely! Total records now: {len(deduped_records)}")
