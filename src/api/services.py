import json
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
from qdrant_client import AsyncQdrantClient
from openai import AsyncOpenAI
from src.config.settings import settings
from src.vector_store.qdrant_indexer import LocalHuggingFaceEmbedder
from src.api.schemas import SearchResultItem, CitationItem


class BaseLLMGenerator(ABC):
    """Abstract Base Class for Vendor-Agnostic LLM Generation Engines."""

    @abstractmethod
    async def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        """Abstract method to generate chat completion responses asynchronously."""
        pass


class OpenAILLMGenerator(BaseLLMGenerator):
    """OpenAI API Async Generator Implementation."""

    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.client = AsyncOpenAI(api_key=api_key)

    async def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content


class RAGService:
    """Async Production RAG Engine Service Layer."""

    _embedder_instance: Optional[LocalHuggingFaceEmbedder] = None

    def __init__(self, generator: Optional[BaseLLMGenerator] = None):
        # Global Singleton Pattern for Heavy Dense Embedding Models
        if RAGService._embedder_instance is None:
            print("[INFO] Initializing Singleton HuggingFace Embedder for API Service...")
            RAGService._embedder_instance = LocalHuggingFaceEmbedder()
        
        self.embedder = RAGService._embedder_instance
        self.async_qdrant_client = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        self.collection_name = "tudelft_academic_publications"
        
        # Pluggable LLM Generator Dependency Injection
        if generator:
            self.generator = generator
        elif settings.OPENAI_API_KEY and not settings.OPENAI_API_KEY.startswith("sk-dummy"):
            self.generator = OpenAILLMGenerator(api_key=settings.OPENAI_API_KEY)
        else:
            self.generator = None

    def _parse_expertise_metadata(self, raw_meta: Any) -> Optional[Dict[str, Any]]:
        """Safely parses dict or JSON string into a valid Pydantic dict."""
        if isinstance(raw_meta, dict):
            return raw_meta
        if isinstance(raw_meta, str) and raw_meta.strip():
            try:
                return json.loads(raw_meta)
            except Exception:
                return None
        return None

    async def search_similar_chunks(
        self, query: str, top_k: int = 3, score_threshold: float = 0.3
    ) -> List[SearchResultItem]:
        """Performs dense vector retrieval asynchronously against Qdrant collection."""
        query_vector = self.embedder.embed_texts([query])[0]
        
        response = await self.async_qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k
        )
        
        points = getattr(response, "points", response)
        results = []

        for hit in points:
            score = getattr(hit, "score", 0.0)
            if score < score_threshold:
                continue

            payload = getattr(hit, "payload", {})
            parsed_meta = self._parse_expertise_metadata(payload.get("expertise_metadata"))

            results.append(
                SearchResultItem(
                    point_id=str(hit.id),
                    similarity_score=round(score, 4),
                    title=payload.get("title", "Untitled"),
                    publication_year=payload.get("publication_year"),
                    is_open_access=payload.get("is_open_access", False),
                    pdf_url=payload.get("pdf_url"),
                    doi=payload.get("doi"),
                    content_text=payload.get("content_text", ""),
                    executive_summary=payload.get("executive_summary"),
                    expertise_metadata=parsed_meta,
                    hyde_questions=payload.get("hyde_questions", [])
                )
            )
        return results

    async def generate_rag_response(
        self, query: str, top_k: int = 3, temperature: float = 0.2
    ) -> Tuple[str, List[CitationItem], int]:
        """Retrieves context and generates an augmented academic answer asynchronously."""
        search_results = await self.search_similar_chunks(query, top_k=top_k, score_threshold=0.2)
        
        if not search_results:
            return "No relevant TU Delft publications found matching your query.", [], 0

        context_blocks = []
        citations_map = {}

        for idx, res in enumerate(search_results, 1):
            context_blocks.append(f"[{idx}] Title: {res.title}\nContent: {res.content_text}")
            citations_map[res.title] = CitationItem(
                openalex_id=res.point_id,
                title=res.title,
                doi=res.doi,
                pdf_url=res.pdf_url,
                publication_year=res.publication_year,
                similarity_score=res.similarity_score
            )

        context_str = "\n\n".join(context_blocks)
        citations_list = list(citations_map.values())

        if not self.generator:
            fallback_text = (
                f"[SYSTEM NOTE: LLM API Key unconfigured].\n"
                f"Retrieved {len(search_results)} relevant context chunks from Qdrant Vector DB.\n\n"
                f"Top Context Snippet:\n{search_results[0].content_text}"
            )
            return fallback_text, citations_list, len(search_results)

        system_prompt = (
            "You are a TU Delft Senior Academic AI Research Assistant. "
            "Answer the user's query accurately using ONLY the provided context snippets. "
            "Cite sources inline using numbers like [1], [2]. If unsure, state that context is insufficient."
        )
        user_prompt = f"Context:\n{context_str}\n\nUser Question: {query}"

        try:
            answer = await self.generator.generate(system_prompt, user_prompt, temperature)
            return answer, citations_list, len(search_results)
        except Exception as e:
            return f"[ERROR] Failed to generate LLM response: {str(e)}", citations_list, len(search_results)


if __name__ == "__main__":
    import asyncio

    async def main():
        service = RAGService()
        res = await service.search_similar_chunks("Majorana Fermions", top_k=1)
        print(f"[SUCCESS] Async RAGService initialized. Retrieved {len(res)} chunk(s).")

    asyncio.run(main())
