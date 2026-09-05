import os
import json
import io
import gzip
import time
import logging
from typing import Dict, Any, Optional
import requests
from azure.storage.blob import BlobServiceClient, ContentSettings

# ==============================================================================
# LOGGING CONFIGURATION
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("TUDelftDataPipeline")

# ==============================================================================
# CONFIGURATION & ENVIRONMENT VARIABLES
# ==============================================================================
AZURE_CONNECTION_STRING: Optional[str] = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME: str = os.getenv("AZURE_CONTAINER_NAME", "tudelft-academic-data")
BLOB_NAME: str = os.getenv("AZURE_BLOB_NAME", "silver/tudelft_oa_works_full.jsonl.gz")

OPENALEX_API_URL: str = "https://api.openalex.org/works"
TU_DELFT_ID: str = "I98358874"
MAILTO: str = os.getenv("OPENALEX_MAILTO", "heriyanto@maranatha.edu")

MAX_RETRIES: int = 5
RETRY_DELAY: int = 3


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def reconstruct_abstract(inverted_index: Optional[Dict[str, Any]]) -> str:
    """
    Reconstructs the full raw text abstract from OpenAlex's abstract_inverted_index.
    
    :param inverted_index: Dictionary mapping words to lists of character positions.
    :return: Reconstructed abstract string.
    """
    if not inverted_index:
        return ""
    
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
            
    word_positions.sort(key=lambda x: x[0])
    return " ".join([word for _, word in word_positions])


def fetch_with_retry(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes API requests with exponential backoff retry logic.
    
    :param params: Dictionary of HTTP query parameters.
    :return: Parsed JSON response from OpenAlex API.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(OPENALEX_API_URL, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed (Attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                sleep_time = RETRY_DELAY * attempt
                logger.info(f"Waiting {sleep_time} seconds before retrying...")
                time.sleep(sleep_time)
            else:
                logger.error("Maximum retry attempts reached.")
                raise e


# ==============================================================================
# MAIN STREAMING LOGIC
# ==============================================================================
def run_safe_azure_stream() -> None:
    """
    Main orchestration function to stream metadata from OpenAlex,
    compress it via GZIP in RAM, and perform a single write transaction to Azure Blob.
    """
    start_time = time.time()
    logger.info("=================================================================")
    logger.info("STARTING PIPELINE: OPENALEX API -> GZIP RAM -> AZURE BLOB STORAGE")
    logger.info("=================================================================\n")

    if not AZURE_CONNECTION_STRING:
        logger.critical("AZURE_STORAGE_CONNECTION_STRING environment variable is not set!")
        raise ValueError("Missing AZURE_STORAGE_CONNECTION_STRING environment variable.")

    logger.info("Connecting to Azure Blob Storage...")
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)

    if not container_client.exists():
        logger.info(f"Container '{CONTAINER_NAME}' does not exist. Creating container...")
        container_client.create_container()

    # In-memory Bytes buffer for compressed streaming
    bytes_buffer = io.BytesIO()
    total_fetched = 0
    page_count = 0
    limit_per_page = 200
    cursor = "*"

    logger.info("Fetching metadata from OpenAlex API into RAM Buffer...")

    # Ensure GzipFile context manager closes cleanly before reading buffer size
    try:
        with gzip.GzipFile(fileobj=bytes_buffer, mode='w') as gz_file:
            while cursor:
                params = {
                    "filter": f"institutions.id:{TU_DELFT_ID},open_access.is_oa:true",
                    "per-page": limit_per_page,
                    "cursor": cursor,
                    "mailto": MAILTO
                }

                try:
                    data = fetch_with_retry(params)
                    results = data.get("results", [])
                    
                    if not results:
                        logger.info("Reached end of OpenAlex dataset!")
                        break

                    for item in results:
                        best_oa = item.get("best_oa_location") or {}
                        
                        record = {
                            "id": item.get("id"),
                            "doi": item.get("doi"),
                            "title": item.get("title"),
                            "publication_year": item.get("publication_year"),
                            "publication_date": item.get("publication_date"),
                            "type": item.get("type"),
                            "cited_by_count": item.get("cited_by_count"),
                            "is_oa": (item.get("open_access") or {}).get("is_oa", True),
                            "pdf_url": best_oa.get("pdf_url"),
                            "landing_page_url": best_oa.get("landing_page_url"),
                            "abstract": reconstruct_abstract(item.get("abstract_inverted_index"))
                        }
                        
                        line = json.dumps(record, ensure_ascii=False) + "\n"
                        gz_file.write(line.encode('utf-8'))

                    total_fetched += len(results)
                    page_count += 1
                    
                    if page_count % 10 == 0:
                        logger.info(f"[PROGRESS] Fetched {total_fetched} records ({page_count} pages completed)...")

                    next_cursor = data.get("meta", {}).get("next_cursor")
                    if next_cursor == cursor:
                        break
                    cursor = next_cursor

                except Exception as e:
                    logger.error(f"Pipeline execution halted at page {page_count} due to error: {e}")
                    logger.info("Flushing collected partial data to GZIP stream for upload...")
                    break
    finally:
        # Guarantee memory pointer inspection happens AFTER Gzip stream closure
        bytes_buffer.seek(0, io.SEEK_END)
        size_mb = bytes_buffer.tell() / (1024 * 1024)

    logger.info(f"Total Records Fetched     : {total_fetched} articles")
    logger.info(f"Compressed RAM Buffer Size: {size_mb:.2f} MB")

    if total_fetched == 0:
        logger.warning("Zero records collected. Execution aborted.")
        return

    # Single-transaction write to Azure Blob Storage
    logger.info(f"Uploading single GZIP blob to '{CONTAINER_NAME}/{BLOB_NAME}'...")
    
    bytes_buffer.seek(0)
    blob_client = container_client.get_blob_client(BLOB_NAME)
    
    blob_client.upload_blob(
        bytes_buffer, 
        overwrite=True,
        content_settings=ContentSettings(
            content_type="application/jsonlines+json",
            content_encoding="gzip"
        )
    )

    elapsed = time.time() - start_time
    logger.info("=================================================================")
    logger.info(f"SUCCESSFULLY UPLOADED TO AZURE BLOB STORAGE IN {elapsed:.2f} SECONDS")
    logger.info("=================================================================")


if __name__ == "__main__":
    run_safe_azure_stream()
