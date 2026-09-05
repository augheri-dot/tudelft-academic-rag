from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        description="Academic search query string",
        json_schema_extra={"example": "Quantum computing advancements"}
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of top context chunks to retrieve"
    )
    score_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity score filter"
    )


class SearchResultItem(BaseModel):
    point_id: str = Field(description="Unique Qdrant UUID point identifier")
    similarity_score: float = Field(description="Cosine similarity score (0.0 - 1.0)")
    title: str = Field(description="Publication title")
    publication_year: Optional[int] = Field(default=None, description="Four-digit publication year")
    is_open_access: bool = Field(default=False, description="Open access accessibility indicator")
    pdf_url: Optional[str] = Field(default=None, description="Direct link to publication PDF stream")
    doi: Optional[str] = Field(default=None, description="Digital Object Identifier")
    content_text: str = Field(description="Matched chunk text segment")
    
    # Enriched Payload Metadata
    executive_summary: Optional[str] = Field(default=None, description="LLM-generated executive summary")
    expertise_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Structured domain expertise")
    hyde_questions: List[str] = Field(default_factory=list, description="Hypothetical user questions")


class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: List[SearchResultItem]


class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        description="User prompt or academic query for RAG answer generation",
        json_schema_extra={"example": "What are the latest breakthroughs in Majorana fermions?"}
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of context chunks to fetch for generation"
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="LLM sampling temperature for response generation"
    )


class CitationItem(BaseModel):
    openalex_id: Optional[str] = Field(default=None, description="Unique source publication ID")
    title: str = Field(description="Publication title")
    doi: Optional[str] = Field(default=None, description="Digital Object Identifier")
    pdf_url: Optional[str] = Field(default=None, description="Direct URL to full-text PDF")
    publication_year: Optional[int] = Field(default=None, description="Publication year")
    similarity_score: Optional[float] = Field(default=None, description="Relevance similarity score")


class ChatResponse(BaseModel):
    query: str
    answer: str
    citations: List[CitationItem]
    retrieved_chunks_count: int


if __name__ == "__main__":
    req = SearchRequest(query="Test Query")
    print("[SUCCESS] API Schemas validated. Sample Request Payload:")
    print(req.model_dump_json(indent=2))
