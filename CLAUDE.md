# CLAUDE.md - AI Assistant Guide for rag-demo

**Last Updated:** 2025-11-21
**Repository:** jprad/rag-demo
**License:** MIT
**Owner:** Jack Radigan

## Project Overview

This repository contains a demonstration of Retrieval-Augmented Generation (RAG) - a pattern that combines information retrieval with language model generation to provide accurate, context-aware responses grounded in specific knowledge bases.

### Current Status

**Development Stage:** Core framework complete, ready for testing
**Primary Branch:** `main`
**Current State:** Full modular RAG framework implemented with Streamlit UI

## Repository Purpose

This is a demonstration project showcasing RAG implementation patterns, including:
- Document ingestion and processing
- Vector embedding generation and storage
- Semantic search capabilities
- Context retrieval and prompt augmentation
- Integration with large language models
- End-to-end RAG pipeline examples

## Codebase Structure

### Recommended Directory Layout

```
rag-demo/
├── src/                    # Source code
│   ├── ingest/            # Document ingestion pipeline
│   ├── embeddings/        # Embedding generation
│   ├── retrieval/         # Vector search and retrieval
│   ├── generation/        # LLM integration
│   ├── pipeline/          # End-to-end RAG pipeline
│   └── utils/             # Shared utilities
├── data/                  # Data directory (gitignored)
│   ├── raw/              # Raw documents
│   ├── processed/        # Processed documents
│   └── vectors/          # Vector stores
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── fixtures/         # Test data
├── notebooks/             # Jupyter notebooks for demos
├── configs/              # Configuration files
├── scripts/              # Utility scripts
├── docs/                 # Documentation
├── .env.example          # Environment variables template
├── requirements.txt      # Python dependencies
├── pyproject.toml        # Python project configuration
└── README.md             # User-facing documentation
```

## Technology Stack

### Expected/Recommended Technologies

**Language:** Python 3.10+ (recommended for AI/ML projects)

**Core RAG Libraries:**
- `langchain` or `llama-index` - RAG framework
- `openai` or `anthropic` - LLM API clients
- `sentence-transformers` or `openai` - Embedding models

**Vector Databases (choose one):**
- `chromadb` - Simple, embedded vector database
- `pinecone-client` - Managed cloud vector database
- `weaviate-client` - Open-source vector database
- `qdrant-client` - High-performance vector search

**Document Processing:**
- `langchain-community` - Document loaders
- `pypdf` or `pdfplumber` - PDF processing
- `python-docx` - Word document processing
- `beautifulsoup4` - HTML processing
- `unstructured` - Multi-format document parsing

**Development Tools:**
- `pytest` - Testing framework
- `black` - Code formatting
- `ruff` or `flake8` - Linting
- `mypy` - Type checking
- `python-dotenv` - Environment management

## Development Workflows

### Initial Setup

When setting up the project for the first time:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Adding New Features

1. **Create feature branch** from main
2. **Implement changes** following conventions below
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Run test suite** to ensure nothing breaks
6. **Commit with descriptive messages**
7. **Push and create PR** for review

### Testing Strategy

**Unit Tests:**
- Test individual functions and classes in isolation
- Mock external dependencies (LLM APIs, vector databases)
- Aim for >80% code coverage

**Integration Tests:**
- Test end-to-end RAG pipelines
- Use smaller test datasets
- May use real services in test mode or local alternatives

**Test Naming Convention:**
```python
def test_<function_name>_<scenario>_<expected_result>():
    # Example: test_embed_documents_empty_list_returns_empty()
    pass
```

### Code Quality Standards

**Python Code Style:**
- Follow PEP 8 conventions
- Use `black` for formatting (line length: 100)
- Use type hints for all function signatures
- Docstrings for all public functions (Google or NumPy style)

**Example:**
```python
from typing import List, Dict, Optional

def retrieve_documents(
    query: str,
    top_k: int = 5,
    filters: Optional[Dict[str, str]] = None
) -> List[Dict[str, any]]:
    """
    Retrieve relevant documents for a given query.

    Args:
        query: The search query text
        top_k: Number of documents to retrieve
        filters: Optional metadata filters

    Returns:
        List of document dictionaries with content and metadata
    """
    pass
```

## Key Conventions

### Configuration Management

- **Never commit secrets or API keys** to the repository
- Use `.env` files for local development (gitignored)
- Provide `.env.example` template with placeholder values
- Use environment variables for all configuration
- Consider `pydantic-settings` for configuration validation

### Data Management

- **Never commit data files** to git
- Add `data/` directory to `.gitignore`
- Document data sources and acquisition steps in README
- Provide sample/synthetic data for testing
- Use version control for data schemas, not data itself

### Error Handling

- Use specific exception types
- Provide informative error messages
- Log errors with appropriate context
- Handle API rate limits and retries gracefully
- Validate inputs early

### Logging

- Use Python `logging` module, not `print()`
- Set appropriate log levels (DEBUG, INFO, WARNING, ERROR)
- Include context in log messages
- Log important events in RAG pipeline:
  - Document ingestion start/completion
  - Embedding generation progress
  - Retrieval queries and results count
  - LLM API calls and token usage

### Performance Considerations

- **Batch operations** when possible (embeddings, database writes)
- **Cache embeddings** to avoid regenerating
- **Implement pagination** for large datasets
- **Use async/await** for I/O-bound operations
- **Monitor API costs** and token usage

## RAG-Specific Best Practices

### Document Chunking

- Choose appropriate chunk sizes (typically 500-1500 tokens)
- Implement overlap between chunks (10-20%)
- Preserve document structure when possible
- Include metadata (source, page, section) with chunks

### Embedding Strategy

- Use consistent embedding model throughout
- Consider domain-specific embedding models
- Cache embeddings to avoid redundant API calls
- Normalize vectors if using cosine similarity

### Retrieval Optimization

- Implement hybrid search (semantic + keyword) when applicable
- Use metadata filters to narrow search space
- Tune similarity thresholds based on use case
- Consider re-ranking results for better relevance

### Prompt Engineering

- Structure prompts consistently
- Include clear instructions for the LLM
- Provide retrieved context explicitly
- Add examples for few-shot learning
- Handle cases where no relevant context is found

### Context Management

- Monitor total token count (retrieval + prompt + generation)
- Implement truncation strategies for long contexts
- Consider summarization for very long documents
- Track and log context window usage

## Security Considerations

### API Key Management

- Store API keys in environment variables only
- Use different keys for development/production
- Rotate keys regularly
- Implement rate limiting to prevent abuse
- Monitor API usage for anomalies

### Data Privacy

- Be aware of data sent to third-party APIs
- Implement data sanitization if needed
- Consider local models for sensitive data
- Document data retention policies
- Comply with relevant regulations (GDPR, etc.)

### Input Validation

- Sanitize user inputs before processing
- Validate file uploads (type, size, content)
- Prevent prompt injection attacks
- Limit query lengths and complexity
- Implement content filtering if needed

## Git Workflow

### Branch Naming

- `feature/<description>` - New features
- `fix/<description>` - Bug fixes
- `docs/<description>` - Documentation updates
- `refactor/<description>` - Code refactoring
- `test/<description>` - Test additions/modifications

### Commit Messages

Follow conventional commit format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(retrieval): add hybrid search with BM25 and vector similarity

Implement hybrid search combining traditional BM25 keyword search
with semantic vector similarity for improved retrieval accuracy.

Closes #42
```

### Pull Request Guidelines

- Provide clear description of changes
- Reference related issues
- Include test coverage information
- Update documentation if needed
- Request review from maintainers
- Ensure CI/CD checks pass

## Common Tasks for AI Assistants

### When Adding New Document Loaders

1. Create loader class in `src/ingest/loaders/`
2. Implement standardized interface (method signatures)
3. Add tests with sample documents
4. Update documentation with supported formats
5. Register loader in factory pattern if applicable

### When Modifying Embedding Pipeline

1. Ensure backward compatibility or migration path
2. Update vector database schema if needed
3. Consider re-embedding existing documents
4. Update tests with new embedding dimensions
5. Document performance implications

### When Updating RAG Pipeline

1. Test with existing queries to prevent regression
2. Benchmark performance (latency, cost, quality)
3. Update prompt templates if needed
4. Validate output format consistency
5. Update integration tests

### When Adding Dependencies

1. Add to `requirements.txt` with version pins
2. Update `README.md` with setup instructions
3. Consider license compatibility
4. Document why dependency is needed
5. Check for security vulnerabilities

## Debugging and Troubleshooting

### Common Issues

**Embedding Dimension Mismatch:**
- Ensure same model used for indexing and querying
- Check vector database configuration
- Verify embedding model version

**Poor Retrieval Quality:**
- Review chunk size and overlap settings
- Check document preprocessing quality
- Tune similarity thresholds
- Validate embedding model selection
- Consider hybrid search approach

**High API Costs:**
- Implement caching strategies
- Batch operations when possible
- Use smaller models for testing
- Monitor token usage
- Consider local model alternatives

**Slow Performance:**
- Profile code to identify bottlenecks
- Implement async operations
- Optimize database queries
- Use batch processing
- Consider caching strategies

### Logging for Debugging

Enable debug logging to troubleshoot issues:

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Resources

### RAG Architecture References

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [Pinecone Learning Center](https://www.pinecone.io/learn/)
- [Vector Database Comparison](https://github.com/erikbern/ann-benchmarks)

### Best Practices

- [Anthropic Prompt Engineering Guide](https://docs.anthropic.com/claude/docs/introduction-to-prompt-design)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [RAG Evaluation Frameworks](https://github.com/explodinggradients/ragas)

## AI Assistant Checklist

When working on this repository, AI assistants should:

- [ ] Review existing code structure before adding new files
- [ ] Follow established naming conventions and patterns
- [ ] Add comprehensive tests for new functionality
- [ ] Update documentation (README, docstrings, this file)
- [ ] Validate configuration and environment setup
- [ ] Check for security issues (API keys, input validation)
- [ ] Monitor token usage and API costs
- [ ] Implement proper error handling and logging
- [ ] Consider performance implications
- [ ] Ensure backward compatibility or provide migration
- [ ] Use type hints and follow code style guidelines
- [ ] Test with sample data before production deployment

## Version History

### v0.2.0 - 2025-11-21 (Core Implementation)
- Full modular RAG framework implemented
- Swappable components: embeddings, vector DB, LLM, chunking, retrieval
- Qdrant vector store with Ollama LLM integration
- Web documentation loaders for 5 IaC tools
- Streamlit chat interface with streaming responses
- CLI tools for ingestion and querying
- Comprehensive configuration system

### v0.1.0 - 2025-11-21 (Initial)
- Repository initialized
- CLAUDE.md created with comprehensive guidelines
- Project structure and conventions defined

---

**Note to AI Assistants:** This document should be updated whenever:
- Project structure changes significantly
- New technologies or patterns are adopted
- Development workflows are modified
- Important decisions or conventions are established

Always keep this file synchronized with the actual state of the repository.
