"""Streamlit chat interface for RAG demo."""

import streamlit as st
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.pipeline.rag import RAGPipeline


# Page configuration
st.set_page_config(
    page_title="RAG Demo - Infrastructure Documentation Assistant",
    page_icon="🤖",
    layout="wide",
)

# Initialize session state
if "messages" in st.session_state:
    messages = st.session_state.messages
else:
    st.session_state.messages = []

if "rag_pipeline" not in st.session_state:
    with st.spinner("Initializing RAG pipeline..."):
        try:
            st.session_state.rag_pipeline = RAGPipeline()
            st.session_state.pipeline_ready = True
        except Exception as e:
            st.error(f"Failed to initialize pipeline: {e}")
            st.session_state.pipeline_ready = False

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")

    st.markdown("### Filters")
    source_filter = st.multiselect(
        "Filter by source",
        options=["ansible", "terraform", "packer", "vagrant", "terraform_provider_esxi"],
        default=None,
    )

    top_k = st.slider("Number of context documents", min_value=1, max_value=10, value=5)

    st.markdown("---")

    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        if st.session_state.get("rag_pipeline"):
            st.session_state.rag_pipeline.clear_history()
        st.rerun()

    st.markdown("---")

    st.markdown("### About")
    st.markdown("""
    This is a RAG (Retrieval-Augmented Generation) demo for infrastructure documentation.

    **Supported Documentation:**
    - Ansible
    - Terraform
    - Packer
    - Vagrant
    - terraform-provider-esxi

    **Tech Stack:**
    - LangChain
    - Qdrant (Vector DB)
    - sentence-transformers
    - Ollama (LLM)
    - Streamlit
    """)

# Main content
st.title("🤖 Infrastructure Documentation Assistant")
st.markdown("Ask questions about Ansible, Terraform, Packer, Vagrant, and more!")

# Check if pipeline is ready
if not st.session_state.get("pipeline_ready", False):
    st.error("⚠️ RAG pipeline is not initialized. Please check your configuration and ensure all services are running.")
    st.markdown("""
    **Requirements:**
    1. Qdrant server running on localhost:6333
    2. Ollama server running on localhost:11434
    3. Documents ingested into the vector database

    Run `python scripts/ingest.py` to ingest documents.
    """)
    st.stop()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show sources if available
        if "sources" in message and message["sources"]:
            with st.expander("📚 View Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {i}** (score: {source['score']:.3f})")
                    st.markdown(f"*From: {source['metadata'].get('source_name', 'unknown')}*")
                    st.text(source["text"])
                    st.markdown("---")

# Chat input
if prompt := st.chat_input("Ask a question about infrastructure documentation"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            # Build filters
            filters = None
            if source_filter:
                filters = {"source_name": source_filter[0]} if len(source_filter) == 1 else None

            # Stream response
            for chunk in st.session_state.rag_pipeline.query_stream(
                question=prompt,
                top_k=top_k,
                filters=filters,
            ):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

            # Get sources (we need to retrieve again for sources)
            response = st.session_state.rag_pipeline.query(
                question=prompt,
                top_k=top_k,
                filters=filters,
            )

            # Show sources
            if response.get("sources"):
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(response["sources"], 1):
                        st.markdown(f"**Source {i}** (score: {source['score']:.3f})")
                        st.markdown(f"*From: {source['metadata'].get('source_name', 'unknown')}*")
                        st.text(source["text"])
                        st.markdown("---")

            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": response.get("sources", [])
            })

        except Exception as e:
            st.error(f"Error generating response: {e}")
            st.markdown("Please ensure Ollama is running and the model is available.")

# Footer
st.markdown("---")
st.markdown(
    "Built with [Streamlit](https://streamlit.io) | "
    "Powered by [LangChain](https://langchain.com), [Qdrant](https://qdrant.tech), and [Ollama](https://ollama.ai)"
)
