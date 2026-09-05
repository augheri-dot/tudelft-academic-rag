import asyncio
import base64
import json
import logging
import os
import random
import re
import sys
import time
import uuid
import zlib
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Any, Optional, List, Tuple
from urllib.parse import urlparse, urljoin

import aiohttp
import pymupdf as fitz
from azure.core.exceptions import AzureError
from azure.storage.blob.aio import BlobServiceClient, BlobClient

# ------------------------------------------------------------------------------
# Logging & Observability Setup
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("GoldPDFExtractionEngine")
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)

# ------------------------------------------------------------------------------
# Infrastructure & Runtime Configuration
# ------------------------------------------------------------------------------
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
INPUT_CONTAINER = os.getenv("AZURE_CONTAINER_NAME", "tudelft-lakehouse")
INPUT_BLOB_PATH = os.getenv("AZURE_BLOB_NAME", "silver/tudelft_oa_works_full.jsonl.gz")
OUTPUT_BLOB_PATH = os.getenv("GOLD_OUTPUT_BLOB_PATH", "gold/tudelft_oa_works_extracted.jsonl.gz")

MAX_CONCURRENT_DOWNLOADS = int(os.getenv("MAX_CONCURRENT_DOWNLOADS", "15"))
CPU_WORKERS = int(os.getenv("CPU_WORKERS", "4"))
QUEUE_MAXSIZE = int(os.getenv("QUEUE_MAXSIZE", "300"))
CHUNK_UPLOAD_SIZE_BYTES = 8 * 1024 * 1024
PDF_PARSE_TIMEOUT_SECONDS = float(os.getenv("PDF_PARSE_TIMEOUT_SECONDS", "15.0"))
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024
STREAM_CHUNK_SIZE_BYTES = 64 * 1024

PDF_MAGIC_BYTES = b"%PDF-"
HTML_META_PDF_REGEX = re.compile(
    r'<meta\s+name=["\'](?:citation_pdf_url|dc.identifier|bepress_citation_pdf_url)["\']\s+content=["\']([^"\']+)["\']',
    re.IGNORECASE
)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0"
]

# ==============================================================================
# MODULE 1: Extraction & URL Resolution Helper
# ==============================================================================
def extract_best_pdf_url(record: Dict[str, Any]) -> Optional[str]:
    if record.get("pdf_url"):
        return record["pdf_url"]

    best_oa = record.get("best_oa_location") or record.get("open_access") or {}
    if isinstance(best_oa, dict):
        if best_oa.get("pdf_url"):
            return best_oa["pdf_url"]
        if best_oa.get("oa_url") and str(best_oa["oa_url"]).lower().endswith(".pdf"):
            return best_oa["oa_url"]

    locations = record.get("locations") or []
    for loc in locations:
        if isinstance(loc, dict):
            if loc.get("pdf_url"):
                return loc["pdf_url"]
            if loc.get("landing_page_url") and str(loc["landing_page_url"]).lower().endswith(".pdf"):
                return loc["landing_page_url"]

    return best_oa.get("oa_url") if isinstance(best_oa, dict) else None


# ==============================================================================
# MODULE 2: High-Performance Isolated CPU Parser Worker
# ==============================================================================
def _parse_pdf_bytes_worker(pdf_bytes: bytes) -> Tuple[Optional[str], str]:
    if not pdf_bytes or len(pdf_bytes) < 10:
        return None, "EMPTY_OR_CORRUPT_PAYLOAD"

    if not pdf_bytes.startswith(PDF_MAGIC_BYTES):
        return None, "INVALID_PDF_HEADER"

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            if doc.is_encrypted:
                if not doc.authenticate(""):
                    return None, "PDF_ENCRYPTED"

            page_count = len(doc)
            if page_count == 0:
                return None, "ZERO_PAGES"

            text_fragments = []
            has_images = False

            for page_num in range(page_count):
                page = doc.load_page(page_num)
                page_text = page.get_text("text").strip()
                if page_text:
                    text_fragments.append(page_text)
                if not has_images and len(page.get_images()) > 0:
                    has_images = True

            full_text = "\n\n".join(text_fragments).strip()
            
            if full_text:
                return full_text, "SUCCESS"
            elif has_images:
                return None, "SCANNED_PDF_NO_TEXT"
            else:
                return None, "EMPTY_TEXT"

    except fitz.FileDataError:
        return None, "CORRUPT_PDF_DATA"
    except Exception as err:
        return None, f"PARSE_ERROR_{type(err).__name__}"


async def parse_pdf_with_timeout(
    executor: ProcessPoolExecutor,
    pdf_bytes: bytes,
    timeout_seconds: float = PDF_PARSE_TIMEOUT_SECONDS
) -> Tuple[Optional[str], str]:
    loop = asyncio.get_running_loop()
    try:
        task = loop.run_in_executor(executor, _parse_pdf_bytes_worker, pdf_bytes)
        return await asyncio.wait_for(task, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        return None, "PARSE_TIMEOUT"
    except Exception as err:
        return None, f"EXECUTION_ERROR_{type(err).__name__}"


# ==============================================================================
# MODULE 3: Enterprise Resilient Downloader & Smart Domain Rate Limiter
# ==============================================================================
class BoundedDomainRateLimiter:
    def __init__(self, limit_per_domain: int = 3, max_domains: int = 1000):
        self.limit_per_domain = limit_per_domain
        self.max_domains = max_domains
        self._semaphores: Dict[str, Tuple[asyncio.Semaphore, float]] = {}

    def get_semaphore(self, url: str) -> asyncio.Semaphore:
        domain = urlparse(url).netloc.lower() or "default"
        now = time.time()

        if len(self._semaphores) > self.max_domains:
            # Purge 20% domain tertua secara cepat
            cutoff_keys = list(self._semaphores.keys())[:int(self.max_domains * 0.2)]
            for k in cutoff_keys:
                del self._semaphores[k]

        if domain not in self._semaphores:
            self._semaphores[domain] = (asyncio.Semaphore(self.limit_per_domain), now)
        else:
            sem, _ = self._semaphores[domain]
            self._semaphores[domain] = (sem, now)

        return self._semaphores[domain][0]


async def fetch_pdf_resilient(
    session: aiohttp.ClientSession,
    url: str,
    rate_limiter: BoundedDomainRateLimiter,
    max_retries: int = 3,
    max_file_size_bytes: int = MAX_FILE_SIZE_BYTES
) -> Tuple[Optional[bytes], str]:
    semaphore = rate_limiter.get_semaphore(url)

    async with semaphore:
        current_target_url = url
        for attempt in range(1, max_retries + 1):
            try:
                timeout = aiohttp.ClientTimeout(
                    total=35.0,
                    connect=8.0,
                    sock_connect=8.0,
                    sock_read=15.0
                )

                headers = {
                    "User-Agent": random.choice(USER_AGENTS),
                    "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9"
                }

                async with session.get(current_target_url, timeout=timeout, allow_redirects=True, headers=headers) as resp:
                    if resp.status in (429, 502, 503, 504):
                        retry_after = resp.headers.get("Retry-After")
                        sleep_duration = float(retry_after) if (retry_after and retry_after.isdigit()) else (attempt * 2.5)
                        await asyncio.sleep(sleep_duration)
                        continue

                    if resp.status != 200:
                        return None, f"HTTP_{resp.status}"

                    content_type = resp.headers.get("Content-Type", "").lower()

                    if any(invalid in content_type for invalid in ("text/html", "application/xhtml+xml")):
                        html_content = await resp.text(errors="ignore")
                        match = HTML_META_PDF_REGEX.search(html_content)
                        if match:
                            extracted_pdf_link = match.group(1).strip()
                            current_target_url = urljoin(str(resp.url), extracted_pdf_link)
                            continue
                        return None, "HTTP_RETURNED_HTML_LANDING_PAGE"

                    buffer = bytearray()
                    async for chunk in resp.content.iter_chunked(STREAM_CHUNK_SIZE_BYTES):
                        buffer.extend(chunk)
                        if len(buffer) > max_file_size_bytes:
                            return None, f"FILE_EXCEEDED_MAX_SIZE_{max_file_size_bytes // (1024*1024)}MB"

                    if not buffer:
                        return None, "EMPTY_PAYLOAD"

                    if not buffer.startswith(PDF_MAGIC_BYTES):
                        return None, "INVALID_BINARY_FORMAT"

                    return bytes(buffer), "HTTP_200"

            except (aiohttp.ClientError, asyncio.TimeoutError) as err:
                if attempt == max_retries:
                    return None, f"NETWORK_ERROR_{type(err).__name__}"
                await asyncio.sleep(attempt * 1.5)

        return None, "RETRIES_EXHAUSTED"


# ==============================================================================
# MODULE 4: Enterprise Azure Block Blob Streaming Writer
# ==============================================================================
class AzureBlockBlobWriter:
    def __init__(self, blob_client: BlobClient):
        self.blob_client = blob_client
        self.block_list: List[str] = []
        self.compressor = zlib.compressobj(level=6, wbits=16 + zlib.MAX_WBITS)
        self.buffer = bytearray()
        self.session_id = str(uuid.uuid4())[:8]
        self.block_counter = 0

    def _generate_block_id(self) -> str:
        raw_id = f"{self.session_id}-block-{self.block_counter:08d}"
        self.block_counter += 1
        return base64.b64encode(raw_id.encode("utf-8")).decode("utf-8")

    async def write_record(self, record: Dict[str, Any]) -> None:
        payload = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        compressed_chunk = self.compressor.compress(payload)

        if compressed_chunk:
            self.buffer.extend(compressed_chunk)

        if len(self.buffer) >= CHUNK_UPLOAD_SIZE_BYTES:
            await self._flush_block()

    async def _flush_block(self, max_retries: int = 3) -> None:
        if not self.buffer:
            return

        block_id = self._generate_block_id()
        data_to_send = bytes(self.buffer)

        for attempt in range(1, max_retries + 1):
            try:
                await self.blob_client.stage_block(block_id=block_id, data=data_to_send)
                self.block_list.append(block_id)
                self.buffer.clear()
                return
            except AzureError as err:
                logger.warning(f"Stage block attempt {attempt}/{max_retries} failed for ID {block_id}: {err}")
                if attempt == max_retries:
                    logger.error(f"Critical failure staging block {block_id}")
                    raise
                await asyncio.sleep(2.0 ** attempt)

    async def finalize(self) -> None:
        try:
            remaining = self.compressor.flush()
            if remaining:
                self.buffer.extend(remaining)

            if self.buffer:
                await self._flush_block()

            if self.block_list:
                logger.info(f"Committing {len(self.block_list)} blocks to Azure Storage...")
                await self.blob_client.commit_block_list(self.block_list)
                logger.info("Successfully committed block list.")
            else:
                logger.warning("No blocks were staged/committed. Output payload was empty.")
        except Exception as err:
            logger.error(f"Failed to finalize Azure Block Blob upload: {err}", exc_info=True)
            raise


# ==============================================================================
# MODULE 5: Resilient Stream Reader & Decompressor
# ==============================================================================
class SmartStreamDecompressor:
    def __init__(self):
        self._decompressor = zlib.decompressobj(32 + zlib.MAX_WBITS)
        self._is_raw = False

    def decompress_chunk(self, chunk: bytes) -> str:
        if self._is_raw:
            return chunk.decode("utf-8", errors="ignore")

        try:
            decompressed = self._decompressor.decompress(chunk)
            return decompressed.decode("utf-8", errors="ignore")
        except zlib.error:
            logger.warning("Stream is not standard Gzip/Zlib binary. Fallback to raw text decoding.")
            self._is_raw = True
            return chunk.decode("utf-8", errors="ignore")

    def flush(self) -> str:
        if self._is_raw:
            return ""
        try:
            remaining = self._decompressor.flush()
            return remaining.decode("utf-8", errors="ignore")
        except zlib.error:
            return ""


# ==============================================================================
# MODULE 6: Production Pipeline Orchestrator
# ==============================================================================
class GoldPDFExtractionEngine:
    def __init__(self):
        self.work_queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
        self.result_queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
        self.process_pool = ProcessPoolExecutor(max_workers=CPU_WORKERS)
        self.rate_limiter = BoundedDomainRateLimiter(limit_per_domain=3)

        self.processed_counter = 0
        self.success_counter = 0
        self.failed_counter = 0
        self.malformed_json_counter = 0

    async def _reader_producer(self, blob_client: BlobClient):
        logger.info(f"Initiating input stream from: {INPUT_BLOB_PATH}")
        decompressor = SmartStreamDecompressor()
        text_buffer = []
        total_records_queued = 0

        try:
            download_stream = await blob_client.download_blob()

            async for chunk in download_stream.chunks():
                decoded_text = decompressor.decompress_chunk(chunk)
                if not decoded_text:
                    continue

                lines = decoded_text.split("\n")
                if text_buffer:
                    lines[0] = "".join(text_buffer) + lines[0]
                    text_buffer.clear()

                text_buffer.append(lines.pop())

                for line in lines:
                    line_str = line.strip()
                    if line_str:
                        try:
                            record = json.loads(line_str)
                            await self.work_queue.put(record)
                            total_records_queued += 1
                        except json.JSONDecodeError:
                            self.malformed_json_counter += 1
                            continue

            final_text = "".join(text_buffer) + decompressor.flush()
            if final_text.strip():
                try:
                    record = json.loads(final_text.strip())
                    await self.work_queue.put(record)
                    total_records_queued += 1
                except json.JSONDecodeError:
                    self.malformed_json_counter += 1

            logger.info(f"Reader producer completed. Queued: {total_records_queued}, Malformed JSON: {self.malformed_json_counter}")

        except Exception as err:
            logger.error(f"Fatal error in _reader_producer: {err}", exc_info=True)
            raise
        finally:
            for _ in range(MAX_CONCURRENT_DOWNLOADS):
                await self.work_queue.put(None)

    async def _async_worker(self, http_session: aiohttp.ClientSession):
        while True:
            record = await self.work_queue.get()
            if record is None:
                self.work_queue.task_done()
                break

            try:
                pdf_url = extract_best_pdf_url(record)
                if not pdf_url:
                    record["full_text"] = None
                    record["extraction_status"] = "NO_URL"
                    await self.result_queue.put((record, False))
                else:
                    pdf_bytes, http_status = await fetch_pdf_resilient(
                        http_session, pdf_url, self.rate_limiter
                    )
                    if not pdf_bytes:
                        record["full_text"] = None
                        record["extraction_status"] = http_status
                        await self.result_queue.put((record, False))
                    else:
                        text, parse_status = await parse_pdf_with_timeout(
                            self.process_pool, pdf_bytes, timeout_seconds=PDF_PARSE_TIMEOUT_SECONDS
                        )
                        record["full_text"] = text
                        record["extraction_status"] = parse_status
                        await self.result_queue.put((record, parse_status == "SUCCESS"))
            except Exception as err:
                logger.error(f"Unexpected error in worker processing record: {err}")
                record["full_text"] = None
                record["extraction_status"] = f"WORKER_ERROR_{type(err).__name__}"
                await self.result_queue.put((record, False))
            finally:
                self.work_queue.task_done()

    async def _writer_consumer(self, blob_writer: AzureBlockBlobWriter):
        while True:
            item = await self.result_queue.get()
            if item is None:
                self.result_queue.task_done()
                break

            record, is_success = item
            self.processed_counter += 1
            if is_success:
                self.success_counter += 1
            else:
                self.failed_counter += 1

            await blob_writer.write_record(record)

            if self.processed_counter % 500 == 0:
                logger.info(
                    f"Progress: Processed={self.processed_counter} | "
                    f"Success={self.success_counter} | Failed={self.failed_counter} | "
                    f"Malformed JSON={self.malformed_json_counter}"
                )

            self.result_queue.task_done()

    async def run(self):
        run_id = str(uuid.uuid4())[:8]
        logger.info(f"Starting Production Lakehouse Pipeline [Run ID: {run_id}]")

        if not AZURE_STORAGE_CONNECTION_STRING:
            logger.error("AZURE_STORAGE_CONNECTION_STRING is missing in environment variables!")
            return

        blob_service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        input_blob = blob_service.get_blob_client(container=INPUT_CONTAINER, blob=INPUT_BLOB_PATH)
        output_blob = blob_service.get_blob_client(container=INPUT_CONTAINER, blob=OUTPUT_BLOB_PATH)

        writer = AzureBlockBlobWriter(output_blob)
        connector = aiohttp.TCPConnector(limit=150, limit_per_host=8, ttl_dns_cache=600)

        async with aiohttp.ClientSession(connector=connector) as session:
            try:
                producer_task = asyncio.create_task(self._reader_producer(input_blob))

                worker_tasks = [
                    asyncio.create_task(self._async_worker(session))
                    for _ in range(MAX_CONCURRENT_DOWNLOADS)
                ]

                writer_task = asyncio.create_task(self._writer_consumer(writer))

                await producer_task
                await asyncio.gather(*worker_tasks)

                await self.result_queue.put(None)
                await writer_task

            except Exception as err:
                logger.error(f"Pipeline Execution Failed: {err}", exc_info=True)
                raise
            finally:
                await writer.finalize()
                self.process_pool.shutdown(wait=True)
                logger.info(
                    f"Pipeline Execution Complete [Run ID: {run_id}]. "
                    f"Total Processed: {self.processed_counter} (Success: {self.success_counter}, Failed: {self.failed_counter})"
                )


if __name__ == "__main__":
    asyncio.run(GoldPDFExtractionEngine().run())
