import sys
from pathlib import Path

# Add the repository root directory to sys.path
root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

# Import main modules after path adjustment


import os
import re
import streamlit as st
from query_rag_rerank import EnterpriseRAGEngineWithRerank

# ---------------------------------------------------------
# DYNAMIC CONFIGURATION (Agnostic & Futureproof Setup)
# ---------------------------------------------------------
APP_TITLE = os.getenv("APP_TITLE", "TU Delft | Academic Intelligence Platform")
APP_SUBTITLE = os.getenv(
    "APP_SUBTITLE",
    "Empowering academic research with automated literature retrieval and verifiable provenance across 10,000+ indexed articles (472k+ vectors)."
)
CHAT_PLACEHOLDER = os.getenv("CHAT_PLACEHOLDER", "Ask a research question about publications...")

# Sample Prompts for Zero-State Onboarding
SUGGESTED_PROMPTS = [
    "What structural analysis techniques and tools are used for computational earthquake management?",
    "How are machine learning algorithms applied to seismic building classification?",
    "What methodologies are used to assess flood risk impact in urban areas?"
]

# 1. Page Configuration
st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS Injection (Clean Styling without Layout Hacks)
st.markdown("""
    <style>
    /* Base Page & Typography */
    .stApp {
        background-color: #FFFFFF;
        color: #0F172A;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Left Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }

    /* Header & Subtitle Styling */
    .main-header {
        font-size: 2rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.25rem;
        letter-spacing: -0.025em;
    }

    .sub-header {
        font-size: 1rem;
        color: #475569;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }

    /* Chat Messages Container & Padding to Avoid Input Overlap */
    .stChatFloatingInputContainer {
        padding-bottom: 1rem !important;
    }

    [data-testid="stChatMessage"] {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }

    [data-testid="stChatMessage"] p {
        color: #0F172A !important;
        font-size: 0.98rem;
        line-height: 1.6;
    }

    /* Hide Default Streamlit Avatars */
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"] {
        display: none !important;
    }

    /* Modern Chat Input Styling */
    [data-testid="stChatInput"] {
        border-radius: 12px !important;
        padding: 4px !important;
    }

    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] input {
        color: #0F172A !important;
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        font-size: 0.95rem !important;
    }

    /* Citation & Reference Links Styling */
    .stMarkdown a {
        color: #2563EB !important;
        text-decoration: underline;
        font-weight: 600;
    }

    /* Suggested Prompts Section */
    .prompt-chip-header {
        font-size: 0.9rem;
        font-weight: 600;
        color: #475569;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_rag_engine():
    """Initializes and caches the Two-Stage RAG Engine."""
    return EnterpriseRAGEngineWithRerank()


try:
    engine = init_rag_engine()
except Exception as e:
    st.error(f"Failed to initialize RAG Core Engine: {str(e)}")
    st.stop()

# 3. Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


def format_response_with_markdown_citations(raw_text: str, sources_metadata=None):
    """
    Parses LLM generated text and formats inline citations into clean numerical 
    markers (e.g., [1], [2]) while compiling a structured references block.
    """
    if not raw_text:
        return "", ""

    # Flexible Regex: Matches [Title](URL or DOI or relative link)
    pattern = r'\[(.*?)\]\(([^)]+)\)'

    ref_markdown_list = []
    url_to_index = {}
    title_to_index = {}

    def normalize_str(s: str) -> str:
        """Helper to normalize strings for robust fuzzy matching."""
        if not s:
            return ""
        return re.sub(r'[\W_]+', '', str(s)).lower()

    # Priority A: Parse structured metadata array from RAG engine
    if sources_metadata and isinstance(sources_metadata, list):
        for idx, item in enumerate(sources_metadata, 1):
            if not isinstance(item, dict):
                continue

            # Standard upright title extraction
            title = str(item.get("title") or "Untitled Article").replace("\n", " ").strip()
            
            # Robust Agnostic Author Parsing
            raw_authors = item.get("authors") or item.get("author") or item.get("creator")
            if isinstance(raw_authors, list):
                formatted_authors = ", ".join([
                    a.get("name", str(a)) if isinstance(a, dict) else str(a) 
                    for a in raw_authors
                ])
            else:
                formatted_authors = str(raw_authors).strip() if raw_authors else ""

            journal = str(item.get("journal") or item.get("publisher") or "").strip()
            year_val = item.get("year") or item.get("published_year") or item.get("date")
            year = f" ({year_val})" if year_val else ""

            raw_url = (
                item.get("url") 
                or item.get("link") 
                or item.get("doi") 
                or item.get("pdf_url") 
                or item.get("source")
            )

            clean_url = ""
            if raw_url is not None:
                str_url = str(raw_url).strip()
                if str_url.lower() not in ["#", "", "none", "null"]:
                    clean_url = f"https://doi.org/{str_url}" if str_url.startswith("10.") else str_url

            meta_str = f" — *{journal}{year}*" if journal else (f" — *{year.strip()}*" if year else "")
            author_str = f" — {formatted_authors}" if formatted_authors else ""

            citation_tag = f"**[{idx}]**"
            
            # Format: **[1]** [Title](URL) — Authors — *Journal (Year)*
            if clean_url:
                url_to_index[clean_url] = citation_tag
                item_md = f"{citation_tag} [{title}]({clean_url}){author_str}{meta_str}"
            else:
                url_to_index[title] = citation_tag
                item_md = f"{citation_tag} {title}{author_str}{meta_str}"

            normalized_title = normalize_str(title)
            if normalized_title:
                title_to_index[normalized_title] = citation_tag

            ref_markdown_list.append(item_md)

    # Priority B: Fallback parsing directly from raw text markdown citations
    else:
        citations = re.findall(pattern, raw_text)
        seen = set()
        for title, url in citations:
            clean_url = url.strip()
            clean_title = title.strip().replace("\n", " ")
            
            if clean_url not in seen and clean_url.lower() not in ["#", "", "none", "null"]:
                seen.add(clean_url)
                idx = len(seen)
                citation_tag = f"**[{idx}]**"
                
                url_to_index[clean_url] = citation_tag
                normalized_title = normalize_str(clean_title)
                if normalized_title:
                    title_to_index[normalized_title] = citation_tag
                
                ref_markdown_list.append(f"{citation_tag} [{clean_title}]({clean_url})")

    # Inline Citation Marker Replacement Engine
    def replace_with_citation_number(match):
        title_text = match.group(1).strip()
        url_text = match.group(2).strip()

        # 1. Match by Exact URL
        if url_text in url_to_index:
            return url_to_index[url_text]

        # 2. Match by DOI string prefix
        if url_text.startswith("10."):
            doi_url = f"https://doi.org/{url_text}"
            if doi_url in url_to_index:
                return url_to_index[doi_url]

        # 3. Match by Normalized Title
        norm_title = normalize_str(title_text)
        if norm_title in title_to_index:
            return title_to_index[norm_title]

        # Fallback: Return original string if no match
        return match.group(0)

    cleaned_text = re.sub(pattern, replace_with_citation_number, raw_text)

    if not ref_markdown_list:
        return cleaned_text, ""

    references_block = "\n\n---\n#### References and Source Material\n" + "\n\n".join(ref_markdown_list)

    return cleaned_text, references_block


# ---------------------------------------------------------
# SIDEBAR CONTROL PANEL
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### Actions")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_prompt = None
        st.rerun()

    st.markdown("---")
    st.markdown("### Retrieval Settings")
    top_n_citations = st.slider("Academic Citations Count", min_value=1, max_value=5, value=3)
    fetch_k_candidates = st.slider("Vector Candidates (HNSW)", min_value=10, max_value=50, value=30)

    st.markdown("---")
    st.markdown("**System Status:** Connected")

    st.markdown("---")
    st.markdown("### Covered Datasets")
    st.markdown("""
    * TU Delft Research Portal
    * OpenAccess Academic Publications
    * Crossref & DOI Metadata Index
    * Peer-Reviewed Journal Articles
    """)

# ---------------------------------------------------------
# MAIN INTERFACE
# ---------------------------------------------------------
st.markdown(f'<div class="main-header">{APP_TITLE}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">{APP_SUBTITLE}</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs([
    "1. RAG Assistant & Literature Retrieval",
    "2. Medallion Storage & Telemetry"
])

# ---------------------------------------------------------
# TAB 1: RAG ASSISTANT & LITERATURE RETRIEVAL
# ---------------------------------------------------------
with tab1:
    st.subheader("Academic Search & Synthesis")

    # Sample Prompt Chips (Zero-State Onboarding)
    if not st.session_state.messages and not st.session_state.pending_prompt:
        st.markdown('<div class="prompt-chip-header">💡 Sample Research Questions:</div>', unsafe_allow_html=True)
        cols = st.columns(len(SUGGESTED_PROMPTS))
        for idx, prompt_text in enumerate(SUGGESTED_PROMPTS):
            with cols[idx]:
                if st.button(prompt_text, key=f"onboarding_chip_{idx}", use_container_width=True):
                    st.session_state.pending_prompt = prompt_text
                    st.rerun()

    # Display Chat History Loop
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar=None):
            if msg["role"] == "assistant":
                cleaned_text, ref_block = format_response_with_markdown_citations(
                    msg.get("content", ""),
                    msg.get("sources_metadata")
                )
                st.markdown(cleaned_text)

                if ref_block:
                    st.markdown(ref_block)
            else:
                st.markdown(msg.get("content", ""))

    # Determine prompt from chat input OR pending prompt chip
    user_input = st.chat_input(CHAT_PLACEHOLDER)
    
    if user_input:
        active_prompt = user_input
        st.session_state.pending_prompt = None
    elif st.session_state.pending_prompt:
        active_prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
    else:
        active_prompt = None

    # Input & Query Handling
    if active_prompt:
        st.session_state.messages.append({"role": "user", "content": active_prompt})

        with st.chat_message("user", avatar=None):
            st.markdown(active_prompt)

        try:
            with st.spinner("Searching and synthesizing academic literature..."):
                result = engine.query(active_prompt, fetch_k=fetch_k_candidates, top_n=top_n_citations)

            if isinstance(result, dict):
                answer_text = result.get("answer", "No response generated.")
                sources_meta = (
                    result.get("sources")
                    or result.get("citations")
                    or result.get("source_documents")
                    or result.get("context")
                )
            else:
                answer_text = str(result)
                sources_meta = None

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_text,
                "sources_metadata": sources_meta
            })

        except Exception as ex:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"The search engine encountered an error: {str(ex)}"
            })

        st.rerun()

# ---------------------------------------------------------
# TAB 2: MEDALLION STORAGE & PIPELINE TELEMETRY
# ---------------------------------------------------------
with tab2:
    st.subheader("Medallion Architecture & Ingestion Telemetry")
    st.caption("Detailed breakdown of asynchronous ETL pipeline execution, binary extraction, and vector index health.")

    # Section 1: Overview Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Scanned Metadata Records", "66,574")
    m2.metric("Gold Full-Text Docs", "11,775", delta="17.7% High Precision")
    m3.metric("Vector Store Scale", "472,000+", delta="Qdrant HNSW")
    m4.metric("Ingestion Rate", "233 rec/sec", delta="1.06 MB/s")

    st.markdown("---")

    # Section 2: Core Engineering Specs & Detailed Status Table
    col_telemetry, col_breakdown = st.columns([1, 1], gap="large")

    with col_telemetry:
        st.markdown("**Gold Layer Extraction Specs**")
        st.json({
            "Execution Time": "286.26 seconds",
            "Data Downloaded": "303.37 MB",
            "Avg Text Payload Length": "63,393 characters/doc",
            "Chunking Strategy": "RecursiveCharacter (500 tokens, 10% overlap)",
            "Primary Storage": "Azure Blob Storage (Gold Parquet Layer)",
            "Vector Database": "Qdrant Cloud (tudelft-academic-rag)"
        })

    with col_breakdown:
        st.markdown("**Extraction Status & Resilience Analysis**")
        st.dataframe(
            [
                {"Status": "SUCCESS (Full-Text Ingested)", "Count": 11775, "Percentage": "17.7%", "Category": "Valid Gold Layer"},
                {"Status": "HTTP 403 (Publisher Paywall)", "Count": 20439, "Percentage": "30.7%", "Category": "External Paywall"},
                {"Status": "NO_URL (Metadata Only)", "Count": 16635, "Percentage": "25.0%", "Category": "Missing Link"},
                {"Status": "HTML Landing Page (No PDF)", "Count": 8489, "Percentage": "12.8%", "Category": "Unresolved Binary"},
                {"Status": "Network Connection Errors", "Count": 5923, "Percentage": "8.9%", "Category": "Graceful Fallback Handling"},
                {"Status": "Invalid Binary Format", "Count": 981, "Percentage": "1.5%", "Category": "Corrupted File"},
                {"Status": "HTTP 418 / Rate Limited", "Count": 545, "Percentage": "0.8%", "Category": "Throttled Request"},
                {"Status": "HTTP 202 / Retries Exhausted", "Count": 748, "Percentage": "1.1%", "Category": "Timeout / Retry Limit"}
            ],
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")
    st.markdown("""
    **Architectural Note:**
    The pipeline processes raw academic metadata from the **Bronze Layer** through **Silver enrichment**, producing **11,775 verified, full-text academic publications** stored in Azure Blob Storage. Over 55% of non-ingested records result from external publisher access controls (HTTP 403) and missing source URLs, which are gracefully categorized without causing pipeline failure.
    """)
