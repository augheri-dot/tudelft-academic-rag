import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cohere
from dotenv import load_dotenv
from openai import OpenAI, APIError
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("EnterpriseRAG_ReRank")


class EnterpriseRAGEngineWithRerank:
    """Enterprise Two-Stage RAG Pipeline featuring Vector Search + Cross-Encoder Re-Ranking."""

    def __init__(
        self,
        collection_name: str = "tudelft_academic_works",
        embedding_model: str = "text-embedding-3-small",
        generation_model: str = "gpt-4o",
        rerank_model: str = "rerank-v3.5",
    ) -> None:
        load_dotenv()
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        self.rerank_model = rerank_model

        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        cohere_api_key = os.getenv("COHERE_API_KEY")

        if not all([qdrant_url, qdrant_api_key, openai_api_key, cohere_api_key]):
            logger.critical("Missing required API keys in .env file.")
            raise ValueError("Environment configuration incomplete.")

        # HTTP REST configuration to bypass gRPC firewall issues
        self.qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            prefer_grpc=False,
            timeout=30.0
        )
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.cohere_client = cohere.ClientV2(api_key=cohere_api_key)

    def _normalize_query(self, query: str) -> str:
        """Sanitizes and normalizes user input query strings."""
        return re.sub(r'\s+', ' ', query).strip()

    def _sanitize_doi(self, raw_doi: Any) -> Tuple[str, str]:
        """Normalizes DOI metadata into both raw identifier and valid HTTP URL."""
        if not raw_doi or str(raw_doi).strip() in ["N/A", "nan", "None"]:
            return "N/A", "N/A"
        
        doi_str = str(raw_doi).strip()
        if doi_str.startswith("http://") or doi_str.startswith("https://"):
            url = doi_str
            raw = doi_str.replace("https://doi.org/", "").replace("http://doi.org/", "")
        else:
            raw = doi_str
            url = f"https://doi.org/{doi_str}"
            
        return raw, url

    def _generate_embedding(self, text: str) -> List[float]:
        """Generates dense vector representation with retry resilience."""
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                response = self.openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=text
                )
                return response.data[0].embedding
            except APIError as e:
                logger.warning(f"OpenAI Embedding attempt {attempt} failed: {e}")
                if attempt == max_retries:
                    raise e
                time.sleep(2 ** attempt)

    def _retrieve_candidates(self, query_vector: List[float], fetch_k: int = 30) -> List[Dict[str, Any]]:
        """Stage 1: Broad Dense HNSW Retrieval with payload selection."""
        max_retries = 2
        search_results = None

        for attempt in range(1, max_retries + 1):
            try:
                search_results = self.qdrant_client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=fetch_k,
                    with_payload=["title", "doi", "text"],
                    search_params={
                        "hnsw_ef": 64,  # Increased ef for higher recall before reranking
                        "exact": False
                    }
                )
                break
            except Exception as e:
                logger.warning(f"Qdrant query attempt {attempt} failed: {e}")
                if attempt == max_retries:
                    logger.error("Max retries reached for Qdrant retrieval pipeline.")
                    raise e
                time.sleep(1)

        candidates = []
        if search_results and hasattr(search_results, 'points'):
            for point in search_results.points:
                payload = point.payload or {}
                raw_doi, doi_url = self._sanitize_doi(payload.get("doi"))
                
                title = payload.get("title")
                if not title or not str(title).strip():
                    title = "Untitled Document"

                candidates.append({
                    "id": point.id,
                    "vector_score": point.score,
                    "title": str(title).strip(),
                    "doi": doi_url,
                    "raw_doi": raw_doi,
                    "text": payload.get("text", "")
                })
        return candidates

    def _rerank_contexts(self, query: str, candidates: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
        """Stage 2: Deduplication & Cross-Encoder Re-Ranking using Cohere v3.5."""
        if not candidates:
            return []

        seen_identifiers = set()
        unique_candidates = []
        for item in candidates:
            identifier = item["doi"] if item["doi"] != "N/A" else item["title"]
            if identifier not in seen_identifiers:
                seen_identifiers.add(identifier)
                unique_candidates.append(item)

        documents = [c["text"] for c in unique_candidates if c.get("text")]
        if not documents:
            return []

        logger.info(f"Re-ranking {len(unique_candidates)} unique candidates down to top {top_n} via {self.rerank_model}...")
        
        try:
            rerank_response = self.cohere_client.rerank(
                model=self.rerank_model,
                query=query,
                documents=documents,
                top_n=min(top_n, len(unique_candidates))
            )
        except Exception as e:
            logger.error(f"Cohere Rerank API call failed: {e}. Falling back to top vector candidates.")
            return unique_candidates[:top_n]

        reranked_results = []
        for result in rerank_response.results:
            original_doc = unique_candidates[result.index]
            original_doc["re_score"] = result.relevance_score
            reranked_results.append(original_doc)

        return reranked_results

    def _build_prompt(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """Constructs a strict grounded prompt with sanitized DOI link instructions."""
        context_str = ""
        for idx, ctx in enumerate(contexts, 1):
            re_score = ctx.get('re_score', 0.0)
            context_str += f"\n--- CONTEXT [{idx}] (Re-Rank Score: {re_score:.4f}) ---\n"
            context_str += f"Title: {ctx['title']}\n"
            context_str += f"DOI Link: {ctx['doi']}\n"
            context_str += f"Content: {ctx['text']}\n"

        system_instruction = (
            "You are an expert academic AI assistant specializing in scientific literature synthesis. "
            "Answer the user query strictly using only the provided highly-relevant academic contexts below. "
            "CRITICAL REQUIREMENT: You must synthesize and incorporate insights from ALL provided context sources "
            "into your response, ensuring multi-source integration rather than relying on just one or two sources. "
            "For every fact or summary statement you make, cite the source using inline Markdown links "
            "formatted as [Title](DOI_URL). Use the exact URL provided in 'DOI Link'. "
            "If no valid DOI Link is available, cite as [Title]. "
            "If the contexts do not contain enough information, state explicitly that the literature lacks sufficient details."
        )

        return f"{system_instruction}\n\nUSER QUERY:\n{query}\n\nRETRIEVED CONTEXTS:\n{context_str}"

    def query(self, user_query: str, fetch_k: int = 30, top_n: int = 3) -> Dict[str, Any]:
        """Executes the complete Two-Stage Synchronous RAG Pipeline."""
        clean_query = self._normalize_query(user_query)
        logger.info(f"Processing normalized query: '{clean_query}'")

        query_vector = self._generate_embedding(clean_query)
        candidates = self._retrieve_candidates(query_vector, fetch_k=fetch_k)
        reranked_contexts = self._rerank_contexts(clean_query, candidates, top_n=top_n)

        if not reranked_contexts:
            return {
                "answer": "No relevant academic literature was found in the database.",
                "sources": []
            }

        prompt = self._build_prompt(clean_query, reranked_contexts)
        response = self.openai_client.chat.completions.create(
            model=self.generation_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )

        return {
            "answer": response.choices[0].message.content,
            "sources": reranked_contexts
        }

    def query_stream(self, user_query: str, fetch_k: int = 30, top_n: int = 3) -> Tuple[Any, List[Dict[str, Any]]]:
        """Executes Two-Stage Retrieval and returns OpenAI Stream Generator + Source Contexts."""
        clean_query = self._normalize_query(user_query)
        logger.info(f"Processing streaming query: '{clean_query}'")

        query_vector = self._generate_embedding(clean_query)
        candidates = self._retrieve_candidates(query_vector, fetch_k=fetch_k)
        reranked_contexts = self._rerank_contexts(clean_query, candidates, top_n=top_n)

        if not reranked_contexts:
            return None, []

        prompt = self._build_prompt(clean_query, reranked_contexts)
        stream = self.openai_client.chat.completions.create(
            model=self.generation_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            stream=True
        )

        return stream, reranked_contexts


if __name__ == "__main__":
    engine = EnterpriseRAGEngineWithRerank()
    test_query = "What are the recent findings on quantum transport and Majorana bound states?"
    result = engine.query(test_query, fetch_k=30, top_n=3)

    print("\n" + "=" * 60)
    print("                RAG ANSWER (WITH RE-RANKING)")
    print("=" * 60)
    print(result["answer"])
