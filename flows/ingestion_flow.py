import sys
import os
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent if CURRENT_DIR.name == "flows" else CURRENT_DIR
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import requests
from typing import List, Dict, Any, Optional

from src.extractors.pdf_pipeline import OpenAlexPDFExtractor
from src.vector_store.qdrant_indexer import QdrantVectorIndexer

OPENALEX_API_URL = os.getenv("OPENALEX_API_URL", "https://api.openalex.org/works")
TU_DELFT_OPENALEX_ID = os.getenv("TU_DELFT_OPENALEX_ID", "I98358874")
DEFAULT_MAILTO = os.getenv("OPENALEX_MAILTO", "adminlibrary@library.tudelft.nl")
DEFAULT_SILVER_PATH = os.getenv("SILVER_DATA_PATH", str(ROOT_DIR / "data" / "silver" / "cleaned_publications.json"))

HTTP_HEADERS = {
    "User-Agent": f"TUDelft-Academic-RAG/1.0 (mailto:{DEFAULT_MAILTO})"
}


def fetch_openalex_publications(
    query: str = "generative ai", 
    limit: int = 10,
    institution_id: str = TU_DELFT_OPENALEX_ID
) -> List[Dict[str, Any]]:
    """Fetches Open Access publication metadata for a given institution from OpenAlex API."""
    print(f"[INFO] Step 1/4: Fetching up to {limit} OA publications from OpenAlex for query: '{query}'...")

    filter_query = f"institutions.id:{institution_id},open_access.is_oa:true"
    if query:
        filter_query += f",title_and_abstract.search:{query}"

    params = {
        "filter": filter_query,
        "per-page": limit,
        "sort": "publication_year:desc",
        "mailto": DEFAULT_MAILTO
    }

    try:
        response = requests.get(OPENALEX_API_URL, params=params, headers=HTTP_HEADERS, timeout=20)
        response.raise_for_status()
        results = response.json().get("results", [])
        print(f"[SUCCESS] Retrieved {len(results)} raw records from OpenAlex.")
        return results
    except requests.RequestException as e:
        print(f"[ERROR] Failed to fetch data from OpenAlex: {e}")
        return []


def run_full_ingestion_pipeline(
    query: str = "generative ai", 
    limit: int = 5,
    collection_name: str = "tudelft_academic_publications",
    silver_path: Optional[str] = None
):
    """Main orchestrator: Ingestion -> Full-Text PDF Extraction -> Chunking -> Vector Indexing."""
    target_silver_path = Path(silver_path) if silver_path else Path(DEFAULT_SILVER_PATH)

    print("\n=======================================================")
    print("STARTING END-TO-END RAG INGESTION PIPELINE")
    print("=======================================================\n")

    # Step 1: Fetch metadata from OpenAlex
    raw_publications = fetch_openalex_publications(query=query, limit=limit)
    if not raw_publications:
        print("[WARN] No records returned from OpenAlex. Terminating pipeline execution.")
        return

    # Step 2: Extract Full-Text PDF content
    print("\n[INFO] Step 2/4: Processing Full-Text PDFs via OpenAlexPDFExtractor...")
    extractor = OpenAlexPDFExtractor()
    processed_publications = []

    for pub in raw_publications:
        try:
            updated_pub = extractor.process_openalex_publication(pub)
            processed_publications.append(updated_pub)
        except Exception as e:
            pub_id = pub.get("id", "Unknown ID")
            print(f"[WARNING] Failed processing PDF for publication {pub_id}: {e}")
            processed_publications.append(pub)

    # Save processed records to Silver Layer storage
    target_silver_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_silver_path, "w", encoding="utf-8") as f:
        json.dump(processed_publications, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved {len(processed_publications)} processed records to Silver Layer: {target_silver_path}")

    # Steps 3 & 4: Chunking and Vector Indexing
    print("\n[INFO] Steps 3/4 & 4/4: Chunking & Indexing into Vector DB...")
    try:
        indexer = QdrantVectorIndexer(collection_name=collection_name)
        indexer.init_collection()

        documents = indexer.prepare_chunks(data_path=str(target_silver_path))
        print(f"[INFO] Total Chunks Generated Across Publications: {len(documents)}")

        if documents:
            indexer.index_documents(documents)
            print("\n=======================================================")
            print("FULL INGESTION PIPELINE COMPLETED SUCCESSFULLY!")
            print("=======================================================\n")
        else:
            print("[WARN] No valid chunks generated for vector indexing.")

    except Exception as e:
        print(f"[ERROR] Ingestion pipeline failed during chunking/indexing phase: {e}")


if __name__ == "__main__":
    run_full_ingestion_pipeline()
