# RAG Demo - Infrastructure Documentation Assistant

A modular Retrieval-Augmented Generation (RAG) system for querying infrastructure-as-code documentation. Built with a fully swappable architecture allowing easy substitution of embedding models, vector databases, LLMs, chunking strategies, and retrieval methods.

## Features

- **Fully Modular Architecture**: Swap any component through configuration
- **Multi-Source Documentation**: Ansible, Terraform, Packer, Vagrant, terraform-provider-esxi
- **Open-Source Stack**: Self-hosted components (Qdrant, Ollama, sentence-transformers)
- **Interactive Chat Interface**: Streamlit-based web UI with streaming responses
- **Configurable Retrieval**: Semantic, hybrid, and keyword-based strategies
- **Flexible Chunking**: Recursive, fixed-size, and paragraph-based strategies

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     RAG Pipeline                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │   Document   │───▶│   Chunking   │───▶│  Embedding   │ │
│  │    Loader    │    │   Strategy   │    │   Provider   │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│         │                                        │          │
│         │                                        ▼          │
│         │                                 ┌──────────────┐ │
│         │                                 │    Vector    │ │
│         │                                 │   Database   │ │
│         │                                 └──────────────┘ │
│         │                                        │          │
│         │                                        │          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │     User     │───▶│  Retrieval   │◀───│              │ │
│  │    Query     │    │   Strategy   │    │              │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│         │                   │                               │
│         │                   │                               │
│         └──────────┬────────┘                               │
│                    ▼                                        │
│             ┌──────────────┐                                │
│             │     LLM      │                                │
│             │   Provider   │                                │
│             └──────────────┘                                │
│                    │                                        │
│                    ▼                                        │
│               ┌─────────┐                                   │
│               │ Response│                                   │
│               └─────────┘                                   │
└─────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component          | Technology                | Swappable Alternatives           |
|--------------------|---------------------------|----------------------------------|
| **RAG Framework**  | LangChain                 | -                                |
| **Vector Database**| Qdrant                    | ChromaDB, Pinecone              |
| **Embeddings**     | sentence-transformers     | OpenAI embeddings               |
| **LLM**            | Ollama (Llama 3, Mistral) | OpenAI GPT, Anthropic Claude    |
| **Web Scraping**   | BeautifulSoup + html2text | Playwright                      |
| **Chunking**       | Recursive                 | Fixed-size, Paragraph, Semantic |
| **Retrieval**      | Semantic/Hybrid           | Keyword, MMR                    |
| **UI**             | Streamlit                 | -                                |

## Prerequisites

### 1. Qdrant Vector Database

**Option A: Docker (Recommended)**
```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 qdrant/qdrant
```

**Option B: Local Installation**
```bash
# Follow instructions at https://qdrant.tech/documentation/quick-start/
```

### 2. Ollama LLM Server

```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3
# or
ollama pull mistral
```

### 3. Python 3.10+

```bash
python --version  # Should be 3.10 or higher
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/jprad/rag-demo.git
cd rag-demo
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env if needed (defaults should work for local setup)
```

### 5. Configure Sources (Optional)

Edit `configs/config.yaml` to:
- Enable/disable specific documentation sources
- Adjust chunking parameters
- Change embedding/LLM models
- Tune retrieval settings

## Usage

### Step 1: Ingest Documentation

Ingest all configured sources:
```bash
python scripts/ingest.py
```

Ingest a specific source:
```bash
python scripts/ingest.py --source ansible
python scripts/ingest.py --source terraform
```

Recreate collection (deletes existing data):
```bash
python scripts/ingest.py --recreate
```

**Note**: Initial ingestion may take 10-30 minutes depending on sources and network speed.

### Step 2: Launch Chat Interface

```bash
streamlit run app.py
```

The web interface will open at `http://localhost:8501`

### Step 3: Ask Questions

Try these example queries:
- "How do I create an Ansible playbook?"
- "What is the difference between Terraform modules and resources?"
- "How do I provision a VM with Packer?"
- "How do I configure Vagrant for multiple machines?"
- "How do I use terraform-provider-esxi to create a virtual machine?"

### Alternative: Command-Line Interface

Single query:
```bash
python scripts/query.py "How do I use Ansible variables?"
```

Interactive mode:
```bash
python scripts/query.py --interactive
```

Filter by source:
```bash
python scripts/query.py --source terraform "How do I create a module?"
```

## Configuration

### Swapping Components

All components can be swapped by editing `configs/config.yaml`:

#### Change Embedding Model

```yaml
embeddings:
  provider: "sentence-transformers"
  config:
    model_name: "all-mpnet-base-v2"  # More accurate but slower
    # or "all-MiniLM-L6-v2"  # Faster but less accurate
```

#### Change Vector Database

```yaml
vector_db:
  provider: "qdrant"  # or "chroma"
  config:
    host: "localhost"
    port: 6333
```

#### Change LLM Model

```yaml
llm:
  provider: "ollama"
  config:
    model: "mistral"  # or "llama3", "codellama", etc.
```

#### Change Chunking Strategy

```yaml
chunking:
  strategy: "recursive"  # or "fixed", "paragraph"
  config:
    chunk_size: 1500
    chunk_overlap: 300
```

#### Change Retrieval Strategy

```yaml
retrieval:
  strategy: "hybrid"  # or "semantic"
  config:
    top_k: 5
    score_threshold: 0.7
```

### Adding New Documentation Sources

Edit `configs/config.yaml`:

```yaml
sources:
  my_new_source:
    url: "https://docs.example.com/"
    enabled: true
    max_depth: 2
    include_patterns: ["/docs/"]
    exclude_patterns: ["/blog/"]
```

Then ingest:
```bash
python scripts/ingest.py --source my_new_source
```

## Project Structure

```
rag-demo/
├── src/                          # Source code
│   ├── embeddings/              # Embedding providers
│   │   ├── providers.py         # SentenceTransformers, OpenAI
│   │   └── factory.py           # Provider factory
│   ├── retrieval/               # Vector stores and retrieval
│   │   ├── vector_stores.py     # Qdrant, ChromaDB
│   │   ├── strategies.py        # Semantic, hybrid retrieval
│   │   └── factory.py           # Retrieval factory
│   ├── generation/              # LLM providers
│   │   ├── llm_providers.py     # Ollama, OpenAI, Anthropic
│   │   └── factory.py           # LLM factory
│   ├── ingest/                  # Document ingestion
│   │   ├── loaders/             # Document loaders
│   │   │   └── web_loader.py    # Web documentation loader
│   │   └── chunking.py          # Chunking strategies
│   ├── pipeline/                # RAG pipelines
│   │   ├── ingestion.py         # Ingestion pipeline
│   │   └── rag.py               # RAG query pipeline
│   └── utils/                   # Utilities
│       ├── config_loader.py     # Configuration management
│       └── interfaces.py        # Abstract base classes
├── configs/                     # Configuration files
│   └── config.yaml              # Main configuration
├── scripts/                     # Utility scripts
│   ├── ingest.py                # Document ingestion script
│   └── query.py                 # Query testing script
├── app.py                       # Streamlit chat interface
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
├── .gitignore                   # Git ignore rules
├── CLAUDE.md                    # AI assistant guidelines
└── README.md                    # This file
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ scripts/ app.py
```

### Linting

```bash
ruff check src/ scripts/ app.py
```

### Type Checking

```bash
mypy src/
```

## Troubleshooting

### Qdrant Connection Error

```
Error: Cannot connect to Qdrant at localhost:6333
```

**Solution**: Ensure Qdrant is running:
```bash
docker ps | grep qdrant  # Check if running
docker start <container-id>  # If stopped
```

### Ollama Model Not Found

```
Error: Model 'llama3' not found
```

**Solution**: Pull the model:
```bash
ollama pull llama3
ollama list  # Verify installation
```

### Slow Embedding Generation

**Solution**: Switch to a smaller model in `configs/config.yaml`:
```yaml
embeddings:
  config:
    model_name: "all-MiniLM-L6-v2"  # Fast and efficient
```

### No Documents Retrieved

**Solution**: Check if documents were ingested:
```bash
python scripts/ingest.py --source ansible
```

### Out of Memory

**Solution**: Reduce batch size or chunk size in `configs/config.yaml`:
```yaml
chunking:
  config:
    chunk_size: 500  # Smaller chunks
```

## Performance Tips

1. **Use GPU for embeddings** (if available):
   ```yaml
   embeddings:
     config:
       device: "cuda"
   ```

2. **Increase chunk overlap** for better context:
   ```yaml
   chunking:
     config:
       chunk_overlap: 300
   ```

3. **Use hybrid retrieval** for better results:
   ```yaml
   retrieval:
     strategy: "hybrid"
   ```

4. **Adjust top_k** based on context window:
   ```yaml
   retrieval:
     config:
       top_k: 5  # More context = better answers
   ```

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Follow existing code style (use `black` for formatting)
2. Add tests for new features
3. Update documentation
4. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LangChain](https://langchain.com) - RAG framework
- [Qdrant](https://qdrant.tech) - Vector database
- [Ollama](https://ollama.ai) - Local LLM inference
- [Sentence Transformers](https://www.sbert.net) - Embedding models
- [Streamlit](https://streamlit.io) - Web interface

## Resources

- [RAG Architecture Guide](https://docs.anthropic.com/claude/docs/retrieval-augmented-generation)
- [Vector Database Comparison](https://github.com/erikbern/ann-benchmarks)
- [Embedding Model Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [Ollama Model Library](https://ollama.ai/library)

---

**Questions or Issues?** Please open an issue on GitHub.
