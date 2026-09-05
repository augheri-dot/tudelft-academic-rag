import logging
import os
import sys
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("EnterpriseRAGEngine")


class EnterpriseRAGEngine:
    """Production-Grade RAG Pipeline with Semantic Retrieval and Citation Guardrails."""

    def __init__(
        self,
        collection_name: str = "tudelft_academic_works",
        embedding_model: str = "text-embedding-3-small",
        generation_model: str = "gpt-4o",
    ) -> None:
        load_dotenv()
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.generation_model = generation_model

        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")

        if not all([qdrant_url, qdrant_api_key, openai_api_key]):
            logger.critical("Missing required API keys or URLs in environment variables.")
            raise ValueError("Environment configuration incomplete.")

        self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=10.0)
        self.openai_client = OpenAI(api_key=openai_api_key)

    def _generate_embedding(self, text: str) -> List[float]:
        """Generates dense vector representation for the input text."""
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise

    def _retrieve_context(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieves top-k relevant text chunks from Qdrant vector database."""
        try:
            search_results = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=True
            )
            
            contexts = []
            for point in search_results.points:
                payload = point.payload or {}
                contexts.append({
                    "id": point.id,
                    "score": point.score,
                    "title": payload.get("title", "Unknown Title"),
                    "doi": payload.get("doi", "N/A"),
                    "text": payload.get("text", "")
                })
            return contexts
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            return []

    def _build_prompt(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """Constructs a strict grounded prompt with embedded context metadata."""
        context_str = ""
        for idx, ctx in enumerate(contexts, 1):
            context_str += f"\n--- CONTEXT [{idx}] ---\n"
            context_str += f"Title: {ctx['title']}\n"
            context_str += f"DOI: {ctx['doi']}\n"
            context_str += f"Content: {ctx['text']}\n"

        system_instruction = (
            "You are an expert academic AI assistant specializing in scientific literature synthesis. "
            "Answer the user query strictly using only the provided academic contexts below. "
            "For every fact, claim, or summary statement you make, you MUST cite the source using inline brackets "
            "containing the document Title and DOI (e.g., [Title, DOI]). "
            "If the contexts do not contain enough information to answer the question, state explicitly that "
            "the available literature does not contain sufficient details."
        )

        return f"{system_instruction}\n\nUSER QUERY:\n{query}\n\nRETRIEVED CONTEXTS:\n{context_str}"

    def query(self, user_query: str, top_k: int = 5) -> Dict[str, Any]:
        """Executes full RAG flow: Embed -> Retrieve -> Synthesize."""
        logger.info(f"Processing query: '{user_query}'")
        
        # 1. Vector Search
        query_vector = self._generate_embedding(user_query)
        contexts = self._retrieve_context(query_vector, top_k=top_k)

        if not contexts:
            return {
                "answer": "No relevant academic literature was found in the database.",
                "sources": []
            }

        # 2. Prompt Engineering & LLM Synthesis
        prompt = self._build_prompt(user_query, contexts)
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.generation_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature to enforce deterministic, grounded output
            )
            answer = response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM Generation failed: {str(e)}")
            answer = "Error generating answer from LLM."

        return {
            "answer": answer,
            "sources": contexts
        }


if __name__ == "__main__":
    engine = EnterpriseRAGEngine()
    
    # Test Query
    test_query = "What are the recent findings on quantum transport and Majorana bound states?"
    result = engine.query(test_query, top_k=3)
    
    print("\n" + "=" * 60)
    print("                     RAG ANSWER")
    print("=" * 60)
    print(result["answer"])
    print("\n" + "=" * 60)
    print("                 RETRIEVED SOURCES")
    print("=" * 60)
    for src in result["sources"]:
        print(f"• [{src['score']:.4f}] {src['title']} ({src['doi']})")
