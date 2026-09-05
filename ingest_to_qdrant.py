"""
Enterprise Qdrant Cloud Ingestion Pipeline for Academic RAG
===========================================================
High-throughput, fault-tolerant, async pipeline to stream Gold Layer documents 
from Azure Blob, chunk text, generate OpenAI embeddings with exponential backoff, 
and batch-upsert into Qdrant Cloud.
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
import zlib
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import AsyncOpenAI, RateLimitError, APIError
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)
from azure.storage.blob.aio import BlobServiceClient

# Load Environment Variables
load_dotenv()

# Logging Config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("EnterpriseQdrantIngestion")

# Env Variables
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME", "tudelft-lakehouse")
BLOB_PATH = os.getenv("GOLD_OUTPUT_BLOB_PATH", "gold/tudelft_oa_works_extracted.jsonl.gz")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "tudelft_academic_works")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = 1536

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
BATCH_SIZE = 128          # Optimum batch size for OpenAI Embeddings
MAX_CONCURRENT_TASKS = 2   # Max parallel OpenAI & Qdrant workers
QUEUE_MAX_SIZE = 50       # Backpressure control for RAM optimization


@dataclass
class IngestionTelemetry:
    processed_documents: int = 0
    skipped_documents: int = 0
    total_chunks_generated: int = 0
    total_vectors_indexed: int = 0
    failed_embeddings: int = 0


class SmartStreamDecompressor:
    """Handles streaming decompress for standard and fallbacked zlib/gzip payloads."""

    def __init__(self):
        self._decompressor = zlib.decompressobj(32 + zlib.MAX_WBITS)
        self._is_raw = False

    def decompress_chunk(self, chunk: bytes) -> str:
        if self._is_raw:
            return chunk.decode("utf-8", errors="ignore")
        try:
            return self._decompressor.decompress(chunk).decode("utf-8", errors="ignore")
        except zlib.error:
            self._is_raw = True
            return chunk.decode("utf-8", errors="ignore")

    def flush(self) -> str:
        if self._is_raw:
            return ""
        try:
            return self._decompressor.flush().decode("utf-8", errors="ignore")
        except zlib.error:
            return ""


class EnterpriseQdrantIngestionEngine:
    def __init__(self):
        if not QDRANT_URL or not QDRANT_API_KEY:
            raise ValueError("QDRANT_URL or QDRANT_API_KEY is missing.")
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is missing.")

        self.qdrant_client = AsyncQdrantClient(
            url=QDRANT_URL, 
            api_key=QDRANT_API_KEY,
            timeout=60.0
        )
        self.openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY, max_retries=3)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.telemetry = IngestionTelemetry()
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)

    async def initialize_collection(self) -> None:
        """Ensures Qdrant collection exists with proper vector index parameters."""
        logger.info(f"Checking Qdrant collection '{COLLECTION_NAME}'...")
        collections = await self.qdrant_client.get_collections()
        existing_names = [c.name for c in collections.collections]

        if COLLECTION_NAME not in existing_names:
            await self.qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Collection '{COLLECTION_NAME}' created successfully.")
        else:
            logger.info(f"Collection '{COLLECTION_NAME}' is ready.")

    async def _generate_embeddings_with_retry(
        self, texts: List[str], max_retries: int = 5
    ) -> List[List[float]]:
        """Generates embeddings with exponential backoff on rate limits/network errors."""
        for attempt in range(1, max_retries + 1):
            try:
                response = await self.openai_client.embeddings.create(
                    input=texts,
                    model=EMBEDDING_MODEL,
                )
                return [data.embedding for data in response.data]
            except (RateLimitError, APIError) as err:
                sleep_time = (2 ** attempt) + (time.time() % 1)
                logger.warning(
                    f"OpenAI API Error (Attempt {attempt}/{max_retries}): {err}. Retrying in {sleep_time:.2f}s..."
                )
                await asyncio.sleep(sleep_time)
            except Exception as err:
                logger.error(f"Non-retryable OpenAI Embedding Error: {err}")
                break

        self.telemetry.failed_embeddings += len(texts)
        return []

    async def _process_batch_worker(self, worker_id: int):
        """Worker task consuming chunk batches and performing embedding + vector indexing."""
        while True:
            batch = await self.queue.get()
            if batch is None:
                self.queue.task_done()
                break

            chunk_ids, payloads, texts = batch
            embeddings = await self._generate_embeddings_with_retry(texts)

            if embeddings:
                points = [
                    PointStruct(id=c_id, vector=emb, payload=pay)
                    for c_id, pay, emb in zip(chunk_ids, payloads, embeddings)
                ]
                try:
                    await self.qdrant_client.upsert(
                        collection_name=COLLECTION_NAME,
                        points=points,
                    )
                    self.telemetry.total_vectors_indexed += len(points)
                except Exception as err:
                    logger.error(f"Worker {worker_id} Qdrant Upsert Error: {err}")

            self.queue.task_done()

    async def run(self) -> None:
        start_time = time.perf_counter()
        await self.initialize_collection()

        # Start Async Worker Pool
        workers = [
            asyncio.create_task(self._process_batch_worker(i))
            for i in range(MAX_CONCURRENT_TASKS)
        ]

        blob_service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        blob_client = blob_service.get_blob_client(container=CONTAINER_NAME, blob=BLOB_PATH)

        if not await blob_client.exists():
            logger.error(f"Target Gold Layer blob does not exist: {BLOB_PATH}")
            return

        logger.info(f"Streaming data from Azure Blob: {CONTAINER_NAME}/{BLOB_PATH}")
        download_stream = await blob_client.download_blob()
        decompressor = SmartStreamDecompressor()

        text_buffer = []
        pending_chunk_ids: List[str] = []
        pending_payloads: List[Dict[str, Any]] = []
        pending_texts: List[str] = []

        async def enqueue_batch():
            nonlocal pending_chunk_ids, pending_payloads, pending_texts
            if pending_texts:
                await self.queue.put((
                    list(pending_chunk_ids),
                    list(pending_payloads),
                    list(pending_texts)
                ))
                pending_chunk_ids.clear()
                pending_payloads.clear()
                pending_texts.clear()

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
                if not line_str:
                    continue

                try:
                    record = json.loads(line_str)
                    full_text = record.get("full_text")
                    status = record.get("extraction_status", "")

                    # Checkpoint Filter
                    if not full_text or "SUCCESS" not in status:
                        self.telemetry.skipped_documents += 1
                        continue

                    self.telemetry.processed_documents += 1

                    # Metadata Extraction
                    doc_id = record.get("id", str(uuid.uuid4()))
                    title = record.get("title") or record.get("display_name", "Untitled Work")
                    publication_year = record.get("publication_year")
                    doi = record.get("doi")
                    pdf_url = record.get("pdf_url") or record.get("primary_location", {}).get("pdf_url")

                    # Chunking
                    chunks = self.text_splitter.split_text(full_text)
                    self.telemetry.total_chunks_generated += len(chunks)

                    for idx, chunk_text in enumerate(chunks):
                        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}_chunk_{idx}"))
                        payload = {
                            "doc_id": doc_id,
                            "chunk_index": idx,
                            "total_chunks": len(chunks),
                            "text": chunk_text,
                            "title": title,
                            "publication_year": publication_year,
                            "doi": doi,
                            "pdf_url": pdf_url,
                            "extraction_status": status,
                        }
                        pending_chunk_ids.append(chunk_id)
                        pending_payloads.append(payload)
                        pending_texts.append(chunk_text)

                        if len(pending_texts) >= BATCH_SIZE:
                            await enqueue_batch()

                    if self.telemetry.processed_documents % 500 == 0:
                        logger.info(
                            f"Progress: Processed Docs={self.telemetry.processed_documents:,} | "
                            f"Indexed Vectors={self.telemetry.total_vectors_indexed:,} | "
                            f"Skipped={self.telemetry.skipped_documents:,}"
                        )

                except json.JSONDecodeError:
                    continue

        # Handle Stream Tail / Remaining Buffer Flush
        final_text = "".join(text_buffer) + decompressor.flush()
        if final_text.strip():
            try:
                record = json.loads(final_text.strip())
                full_text = record.get("full_text")
                status = record.get("extraction_status", "")
                if full_text and "SUCCESS" in status:
                    self.telemetry.processed_documents += 1
                    doc_id = record.get("id", str(uuid.uuid4()))
                    title = record.get("title") or record.get("display_name", "Untitled Work")
                    chunks = self.text_splitter.split_text(full_text)
                    self.telemetry.total_chunks_generated += len(chunks)
                    for idx, chunk_text in enumerate(chunks):
                        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}_chunk_{idx}"))
                        payload = {
                            "doc_id": doc_id,
                            "chunk_index": idx,
                            "total_chunks": len(chunks),
                            "text": chunk_text,
                            "title": title,
                            "publication_year": record.get("publication_year"),
                            "doi": record.get("doi"),
                            "pdf_url": record.get("pdf_url") or record.get("primary_location", {}).get("pdf_url"),
                            "extraction_status": status,
                        }
                        pending_chunk_ids.append(chunk_id)
                        pending_payloads.append(payload)
                        pending_texts.append(chunk_text)
            except json.JSONDecodeError:
                pass

        # Flush Remaining Queue
        await enqueue_batch()

        # Wait for all background tasks to finish
        await self.queue.join()

        # Stop Workers
        for _ in range(MAX_CONCURRENT_TASKS):
            await self.queue.put(None)
        await asyncio.gather(*workers)

        await blob_service.close()
        await self.qdrant_client.close()

        elapsed = time.perf_counter() - start_time
        logger.info("=" * 60)
        logger.info("         ENTERPRISE QDRANT INGESTION TELEMETRY          ")
        logger.info("=" * 60)
        logger.info(f"Execution Time          : {elapsed:.2f} seconds")
        logger.info(f"Processed Documents     : {self.telemetry.processed_documents:,}")
        logger.info(f"Skipped Documents       : {self.telemetry.skipped_documents:,}")
        logger.info(f"Total Chunks Generated  : {self.telemetry.total_chunks_generated:,}")
        logger.info(f"Total Vectors Indexed   : {self.telemetry.total_vectors_indexed:,}")
        logger.info(f"Failed Embeddings       : {self.telemetry.failed_embeddings:,}")
        logger.info("=" * 60)


if __name__ == "__main__":
    engine = EnterpriseQdrantIngestionEngine()
    asyncio.run(engine.run())
