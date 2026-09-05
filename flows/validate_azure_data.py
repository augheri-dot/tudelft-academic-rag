import os
import sys
import json
import gzip
import io
import logging
import math
from datetime import datetime, timezone
from typing import Dict, Any, Generator, Optional, List, Set
from pydantic import BaseModel, Field, ValidationError
from azure.storage.blob import BlobServiceClient, StorageStreamDownloader
from azure.core.exceptions import AzureError

# ==============================================================================
# STRUCTURED JSON LOGGING SYSTEM (Datadog / OpenTelemetry Ready)
# ==============================================================================
class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)

logger = logging.getLogger("EnterpriseDataValidationEngine")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

# ==============================================================================
# PYDANTIC DATA CONTRACT SCHEMA ENFORCEMENT
# ==============================================================================
class WorkRecordSchema(BaseModel):
    id: str = Field(..., min_length=5)
    title: Optional[str] = None
    publication_year: Optional[int] = Field(None, ge=1600, le=2100)
    doi: Optional[str] = None
    pdf_url: Optional[str] = None
    abstract: Optional[str] = None
    language: Optional[str] = None
    cited_by_count: Optional[int] = Field(0, ge=0)

# ==============================================================================
# DATA QUALITY AUDIT ENGINE
# ==============================================================================
class DataQualityAuditEngine:
    def __init__(self):
        self.connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.container_name = os.getenv("AZURE_CONTAINER_NAME", "tudelft-lakehouse")
        self.blob_name = os.getenv("AZURE_BLOB_NAME", "silver/tudelft_oa_works_full.jsonl.gz")
        
        if not self.connection_string:
            logger.critical("AZURE_STORAGE_CONNECTION_STRING environment variable is unset.")
            raise ValueError("Missing Connection String")

        self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)

    def _stream_lines_stateless(self, downloader: StorageStreamDownloader) -> Generator[str, None, None]:
        """
        Stateless streaming line generator. 
        Supports both raw JSONL and GZIP bytes directly from network buffer.
        """
        stream = downloader.chunks()
        buffer = bytearray()
        is_gzip = None

        for chunk in stream:
            buffer.extend(chunk)

            # Detect compression type on first chunk
            if is_gzip is None and len(buffer) >= 2:
                is_gzip = (buffer[0] == 0x1f and buffer[1] == 0x8b)

            if is_gzip:
                try:
                    with gzip.GzipFile(fileobj=io.BytesIO(buffer), mode='rb') as gz:
                        decompressed = gz.read()
                        lines = decompressed.splitlines()
                        for line in lines[:-1]:
                            yield line.decode('utf-8')
                        buffer = bytearray(lines[-1]) if lines else bytearray()
                except (gzip.BadGzipFile, EOFError):
                    continue
            else:
                while b'\n' in buffer:
                    line, _, remaining = buffer.partition(b'\n')
                    buffer = bytearray(remaining)
                    yield line.decode('utf-8')

        if buffer:
            if is_gzip:
                try:
                    with gzip.GzipFile(fileobj=io.BytesIO(buffer), mode='rb') as gz:
                        yield gz.read().decode('utf-8')
                except Exception:
                    pass
            else:
                yield buffer.decode('utf-8')

    def _calculate_entropy(self, text: str) -> float:
        """Calculates Shannon Entropy to filter junk/repetitive text."""
        if not text:
            return 0.0
        prob = [float(text.count(c)) / len(text) for c in set(text)]
        return - sum([p * math.log(p, 2) for p in prob])

    def run_audit((self) -> Dict[str, Any]:
        logger.info(f"Initiating streaming audit for blob: '{self.blob_name}'")
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name, 
            blob=self.blob_name
        )

        if not blob_client.exists():
            logger.error(f"Target blob '{self.blob_name}' does not exist.")
            return {}

        downloader = blob_client.download_blob(max_concurrency=4)

        # Metrics initialization
        total_records = 0
        duplicate_records = 0
        schema_violations = 0
        
        abstract_valid = 0
        abstract_missing = 0
        abstract_short = 0
        low_entropy_abstracts = 0
        
        has_title = 0
        has_doi = 0
        has_pdf = 0
        
        seen_ids: Set[str] = set()
        publication_years: Dict[int, int] = {}
        sample_records: List[Dict[str, Any]] = []

        # Execute Stateless Line Processing
        for line in self._stream_lines_stateless(downloader):
            line_clean = line.strip()
            if not line_clean:
                continue

            total_records += 1
            
            try:
                raw_json = json.loads(line_clean)
                record = WorkRecordSchema(**raw_json)
            except (json.JSONDecodeError, ValidationError):
                schema_violations += 1
                continue

            # Deduplication Check
            if record.id in seen_ids:
                duplicate_records += 1
            else:
                seen_ids.add(record.id)

            # Metadata Completeness
            if record.title:
                has_title += 1
            if record.doi:
                has_doi += 1
            if record.pdf_url:
                has_pdf += 1

            # RAG Text Quality & Entropy Check
            if not record.abstract:
                abstract_missing += 1
            elif len(record.abstract.strip()) < 50:
                abstract_short += 1
            else:
                entropy = self._calculate_entropy(record.abstract)
                if entropy < 3.0: # Filter repetitive placeholder text
                    low_entropy_abstracts += 1
                else:
                    abstract_valid += 1

            # Temporal Stats
            if record.publication_year:
                publication_years[record.publication_year] = publication_years.get(record.publication_year, 0) + 1

            # Capture Samples
            if len(sample_records) < 2:
                sample_records.append(record.model_dump())

        # Enterprise Weighted Quality Score Calculation
        if total_records > 0:
            weight_abstract = 0.45
            weight_title = 0.25
            weight_doi = 0.15
            weight_pdf = 0.15
            
            score_abstract = (abstract_valid / total_records) * 100
            score_title = (has_title / total_records) * 100
            score_doi = (has_doi / total_records) * 100
            score_pdf = (has_pdf / total_records) * 100
            
            penalty_schema = (schema_violations / total_records) * 100

            composite_dq_score = (
                (score_abstract * weight_abstract) +
                (score_title * weight_title) +
                (score_doi * weight_doi) +
                (score_pdf * weight_pdf)
            ) - penalty_schema
            composite_dq_score = max(0.0, min(100.0, composite_dq_score))
        else:
            composite_dq_score = 0.0

        # Construct Report Payload
        report = {
            "audit_metadata": {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "target_container": self.container_name,
                "target_blob": self.blob_name,
                "engine_version": "2.0.0-enterprise"
            },
            "summary_metrics": {
                "total_records_processed": total_records,
                "unique_records": len(seen_ids),
                "duplicate_records": duplicate_records,
                "schema_violations": schema_violations,
                "composite_data_quality_score": round(composite_dq_score, 2)
            },
            "rag_readiness_metrics": {
                "valid_high_quality_abstracts": abstract_valid,
                "missing_abstracts": abstract_missing,
                "short_abstracts": abstract_short,
                "low_entropy_junk_abstracts": low_entropy_abstracts,
                "title_completeness_pct": round((has_title / total_records * 100), 2) if total_records else 0,
                "doi_availability_pct": round((has_doi / total_records * 100), 2) if total_records else 0,
                "pdf_url_availability_pct": round((has_pdf / total_records * 100), 2) if total_records else 0
            },
            "temporal_distribution_top_5": dict(
                sorted(publication_years.items(), key=lambda x: x[0], reverse=True)[:5]
            )
        }

        # Console Output Print
        self._print_console_report(report)

        # Persist Report Artifact to Azure Blob Storage
        self._persist_report_to_azure(report)

        return report

    def _print_console_report(self, report: Dict[str, Any]):
        print("\n" + "="*75)
        print("         ENTERPRISE DATA QUALITY AUDIT REPORT (WORLD-CLASS)        ")
        print("="*75)
        print(f" Timestamp (UTC)         : {report['audit_metadata']['timestamp_utc']}")
        print(f" Total Audited Records   : {report['summary_metrics']['total_records_processed']:,}")
        print(f" Unique Records          : {report['summary_metrics']['unique_records']:,}")
        print(f" Schema Violations       : {report['summary_metrics']['schema_violations']:,}")
        print(f" Composite DQ Score      : {report['summary_metrics']['composite_data_quality_score']}%")
        print("-" * 75)
        print(" RAG READINESS METRICS:")
        print(f"  - Valid Abstracts      : {report['rag_readiness_metrics']['valid_high_quality_abstracts']:,}")
        print(f"  - Missing Abstracts    : {report['rag_readiness_metrics']['missing_abstracts']:,}")
        print(f"  - Low-Entropy (Junk)   : {report['rag_readiness_metrics']['low_entropy_junk_abstracts']:,}")
        print(f"  - Title Completeness   : {report['rag_readiness_metrics']['title_completeness_pct']}%")
        print(f"  - DOI Completeness     : {report['rag_readiness_metrics']['doi_availability_pct']}%")
        print(f"  - Direct PDF Links     : {report['rag_readiness_metrics']['pdf_url_availability_pct']}%")
        print("="*75 + "\n")

    def _persist_report_to_azure(self, report: Dict[str, Any]):
        """Persists structured report artifact back to Azure Storage."""
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        report_blob_name = f"silver/audit_reports/dq_report_{date_str}.json"
        
        logger.info(f"Persisting JSON audit artifact to Azure: '{report_blob_name}'")
        try:
            report_client = self.blob_service_client.get_blob_client(
                container=self.container_name, 
                blob=report_blob_name
            )
            report_bytes = json.dumps(report, indent=2).encode('utf-8')
            report_client.upload_blob(report_bytes, overwrite=True)
            logger.info("Successfully persisted audit report artifact.")
        except AzureError as e:
            logger.error(f"Failed to persist audit artifact to Azure: {str(e)}")

if __name__ == "__main__":
    engine = DataQualityAuditEngine()
    engine.run_audit()
