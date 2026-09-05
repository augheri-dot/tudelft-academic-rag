import json
from abc import ABC, abstractmethod
from typing import List, Optional
from pydantic import BaseModel, Field
from openai import OpenAI, APIError, AuthenticationError, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from src.config.settings import settings


class ExpertiseMetadata(BaseModel):
    primary_domain: str = Field(
        default="General Academic",
        description="Primary academic or engineering domain (e.g., Civil Engineering, Quantum Computing)"
    )
    methodologies: List[str] = Field(
        default_factory=list,
        description="List of specific methodologies, algorithms, or analytical frameworks used"
    )
    core_tools: List[str] = Field(
        default_factory=list,
        description="List of software, hardware, computational libraries, or instruments utilized"
    )


class LLMEnrichmentOutput(BaseModel):
    reconstructed_text: str = Field(
        description="Cleaned full-text/abstract stripped of PDF artifacts, headers, and footers"
    )
    executive_summary: str = Field(
        description="Concise 2-3 sentence executive business-friendly summary"
    )
    expertise: ExpertiseMetadata = Field(
        default_factory=ExpertiseMetadata,
        description="Structured domain expertise and technical capabilities"
    )
    hyde_questions: List[str] = Field(
        default_factory=list,
        description="2-3 hypothetical user queries directly answered by this publication"
    )


class BaseLLMEnricher(ABC):
    """Abstract Base Class for Vendor-Agnostic LLM Enrichment Engines."""

    @abstractmethod
    def enrich_publication(self, title: str, raw_text: str) -> Optional[LLMEnrichmentOutput]:
        """Abstract method to enrich raw academic text into structured metadata."""
        pass


class OpenAIEnricher(BaseLLMEnricher):
    """OpenAI Implementation for Structured Academic Document Enrichment."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.api_key = settings.OPENAI_API_KEY
        self.client = OpenAI(api_key=self.api_key) if self._is_key_valid() else None

    def _is_key_valid(self) -> bool:
        return bool(self.api_key and not self.api_key.startswith("sk-dummy"))

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((RateLimitError, APIError))
    )
    def _execute_completion(self, prompt: str) -> str:
        """Executes API completion with exponential backoff for transient rate limits/API errors."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a senior data engineer extracting structured academic expertise."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        return response.choices[0].message.content

    def enrich_publication(self, title: str, raw_text: str) -> Optional[LLMEnrichmentOutput]:
        """Extracts executive summaries, HyDE questions, and expertise schemas from input text."""
        if not raw_text or not self.client:
            print("[WARNING] OpenAI API key is unconfigured or input text is empty. Bypassing LLM enrichment.")
            return None

        prompt = f"""Process the following academic text from TU Delft publication:

TITLE: {title}
RAW TEXT SNIPPET: {raw_text[:4000]}

Respond ONLY with a JSON object strictly matching this schema:
{{
  "reconstructed_text": "Cleaned version of the raw text removing PDF noise, headers, and footers.",
  "executive_summary": "2-3 sentences explaining business value and practical application.",
  "expertise": {{
    "primary_domain": "Core domain name",
    "methodologies": ["Method 1", "Method 2"],
    "core_tools": ["Tool 1", "Tool 2"]
  }},
  "hyde_questions": [
    "Hypothetical question 1 that this paper answers?",
    "Hypothetical question 2?"
  ]
}}"""

        try:
            print(f"[INFO] Invoking {self.model_name} enrichment for: '{title[:45]}...'")
            json_payload = self._execute_completion(prompt)
            parsed_dict = json.loads(json_payload)
            
            validated_output = LLMEnrichmentOutput(**parsed_dict)
            print(f"[SUCCESS] LLM Enrichment executed successfully for: '{title[:45]}...'")
            return validated_output

        except AuthenticationError:
            print("[ERROR] OpenAI Authentication failed. Verify your OPENAI_API_KEY environment variable.")
            return None
        except Exception as e:
            print(f"[ERROR] LLM Enrichment encountered an unhandled failure: {str(e)}")
            return None


if __name__ == "__main__":
    enricher = OpenAIEnricher()
    sample_title = "The Circular Economy - A new sustainability paradigm"
    sample_text = "The circular economy is an economic system aimed at eliminating waste. We propose a framework using life cycle assessment and material flow analysis."
    
    output = enricher.enrich_publication(sample_title, sample_text)
    if output:
        print("\n--- ENRICHMENT OUTPUT MATRIX ---")
        print(json.dumps(output.model_dump(), indent=2))
