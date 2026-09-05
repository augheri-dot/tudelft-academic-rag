import json
import os
import uuid
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from src.config.settings import settings


class BaseEmbedder(ABC):
    """Abstract Base Class for Vendor-Agnostic Dense Embedding Models."""

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generates dense vector embeddings for input strings."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Returns vector output dimension size."""
        pass


class LocalHuggingFaceEmbedder(BaseEmbedder):
    """HuggingFace Sentence Transformers Embedder Implementation."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"[INFO] Initializing Dense Embedding Model: '{model_name}'...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_embedding_dimension()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def get_dimension(self) -> int:
        return self.dimension


class QdrantVectorIndexer:
    """Vector Store Manager for Qdrant Hybrid Payload Indexing."""

    def __init__(
        self,
        collection_name: str = "tudelft_academic_publications",
        embedder: Optional[BaseEmbedder] = None
    ):
        self.collection_name = collection_name
        self.embedder = embedder or LocalHuggingFaceEmbedder()
        self.client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

    def init_collection(self):
        """Initializes or resets Qdrant Collection with dynamic embedding dimensions."""
        collections = [c.name for c in self.client.get_collections().collections]
        vector_dim = self.embedder.get_dimension()

        if self.collection_name in collections:
            print(f"[INFO] Collection '{self.collection_name}' already exists. Recreating...")
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE)
        )
        print(f"[SUCCESS] Initialized collection '{self.collection_name}' (Vector Dim: {vector_dim}).")

    def _recursive_chunk_text(
        self, text: str, chunk_size: int = 800, overlap: int = 150
    ) -> List[str]:
        """Splits text along natural structural boundaries (paragraphs, sentences, words)."""
        if not text:
            return []

        separators = ["\n\n", "\n", ". ", " ", ""]
        final_chunks = []

        def _split_recursive(text_segment: str, sep_idx: int):
            if len(text_segment) <= chunk_size or sep_idx >= len(separators):
                if text_segment.strip():
                    final_chunks.append(text_segment.strip())
                return

            sep = separators[sep_idx]
            splits = text_segment.split(sep) if sep else list(text_segment)
            current_chunk = ""

            for split in splits:
                item = split + sep if sep else split
                if len(current_chunk) + len(item) <= chunk_size:
                    current_chunk += item
                else:
                    if current_chunk:
                        _split_recursive(current_chunk, sep_idx + 1)
                    current_chunk = item

            if current_chunk:
                _split_recursive(current_chunk, sep_idx + 1)

        _split_recursive(text, 0)

        # Apply simple sliding window overlap
        overlapped_chunks = []
        for i, chunk in enumerate(final_chunks):
            if i > 0 and overlap > 0:
                prev_overlap = final_chunks[i - 1][-overlap:]
                overlapped_chunks.append(prev_overlap + " " + chunk)
            else:
                overlapped_chunks.append(chunk)

        return overlapped_chunks

    def prepare_chunks(self, data_path: str = "data/silver/cleaned_publications.json") -> List[Dict[str, Any]]:
        """Parses Silver Layer data, chunks text, and constructs payload records with UUID keys."""
        if not os.path.exists(data_path):
            print(f"[WARNING] Data path '{data_path}' not found.")
            return []

        with open(data_path, "r", encoding="utf-8") as f:
            publications = json.load(f)

        documents = []
        namespace_uuid = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

        for pub in publications:
            openalex_id = pub.get("openalex_id", "")
            title = pub.get("title", "")
            content = pub.get("full_text") or pub.get("reconstructed_abstract") or ""

            base_content = f"Title: {title}\nSummary: {pub.get('executive_summary', '')}\nContent: {content}".strip()
            if not base_content:
                continue

            chunks = self._recursive_chunk_text(base_content, chunk_size=800, overlap=150)

            for chunk_idx, chunk in enumerate(chunks):
                # Deterministic UUID v5 for idempotent upserts
                unique_key = f"{openalex_id}_chunk_{chunk_idx}"
                point_uuid = str(uuid.uuid5(namespace_uuid, unique_key))

                documents.append({
                    "point_id": point_uuid,
                    "text": chunk,
                    "payload": {
                        "openalex_id": openalex_id,
                        "title": title,
                        "doi": pub.get("doi"),
                        "pdf_url": pub.get("pdf_url"),
                        "publication_year": pub.get("publication_year"),
                        "is_open_access": pub.get("is_open_access", False),
                        "cited_by_count": pub.get("cited_by_count", 0),
                        "executive_summary": pub.get("executive_summary"),
                        "expertise_metadata": pub.get("expertise_metadata"),
                        "hyde_questions": pub.get("hyde_questions", []),
                        "chunk_index": chunk_idx,
                        "content_text": chunk
                    }
                })

        return documents

    def index_documents(self, documents: List[Dict[str, Any]]):
        """Generates dense vector embeddings and upserts chunk points to Qdrant."""
        if not documents:
            print("[WARNING] No documents available for vector indexing.")
            return

        texts = [doc["text"] for doc in documents]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedder.embed_texts(texts)

        points = []
        for doc, emb in zip(documents, embeddings):
            points.append(
                PointStruct(
                    id=doc["point_id"],
                    vector=emb,
                    payload=doc["payload"]
                )
            )

        print(f"[INFO] Upserting {len(points)} vector points into Qdrant...")
        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"[SUCCESS] Indexed {len(points)} vector points into Qdrant Vector DB!")


if __name__ == "__main__":
    indexer = QdrantVectorIndexer()
    indexer.init_collection()
    docs = indexer.prepare_chunks()
    indexer.index_documents(docs)
