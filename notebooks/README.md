# RAG Demo Notebooks

This directory contains Jupyter notebooks demonstrating the RAG system's features and capabilities.

## Notebooks

### 1. [01_quick_start.ipynb](01_quick_start.ipynb)
**Level:** Beginner
**Duration:** 10-15 minutes

A quick introduction to the RAG system covering:
- Basic document ingestion
- Simple queries
- Understanding retrieval and generation
- Metadata filtering

**Prerequisites:**
- Qdrant running on localhost:6333
- Ollama with llama3 model
- Dependencies installed

### 2. [02_advanced_features.ipynb](02_advanced_features.ipynb)
**Level:** Intermediate
**Duration:** 20-30 minutes

Explores advanced RAG features:
- Hybrid retrieval (semantic + keyword)
- Different chunking strategies
- Streaming responses
- Performance tuning
- Examining retrieved context

**Prerequisites:**
- Completed quick start notebook
- Understanding of RAG basics

## Getting Started

### 1. Install Jupyter

```bash
pip install jupyter notebook
# or
pip install jupyterlab
```

### 2. Start Services

Ensure required services are running:

```bash
# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Verify Ollama is running
ollama list
```

### 3. Launch Jupyter

```bash
# From the project root
jupyter notebook notebooks/

# or for JupyterLab
jupyter lab notebooks/
```

### 4. Run Notebooks

Open a notebook and run cells sequentially using:
- `Shift + Enter` - Run cell and move to next
- `Ctrl + Enter` - Run cell and stay
- `Alt + Enter` - Run cell and insert new cell below

## Tips

1. **Run cells in order** - Notebooks are designed to be executed sequentially
2. **Check outputs** - Verify each cell's output before proceeding
3. **Modify examples** - Experiment with different queries and parameters
4. **Resource usage** - Some cells may take time if running on CPU

## Troubleshooting

### Import Errors

```python
import sys
sys.path.append('..')  # Ensure this is in your first cell
```

### Service Connection Errors

Verify services are running:
```bash
# Check Qdrant
curl http://localhost:6333/health

# Check Ollama
curl http://localhost:11434/api/tags
```

### Out of Memory

Reduce batch sizes or use smaller models:
```yaml
embeddings:
  config:
    model_name: "all-MiniLM-L6-v2"  # Smaller, faster model
```

## Additional Resources

- [Main README](../README.md) - Project overview and setup
- [CLAUDE.md](../CLAUDE.md) - Development guidelines
- [LangChain Docs](https://python.langchain.com/docs/get_started/introduction)
- [Qdrant Docs](https://qdrant.tech/documentation/)

## Contributing

To add a new notebook:

1. Create notebook in this directory
2. Follow naming convention: `NN_descriptive_name.ipynb`
3. Update this README with description
4. Test notebook end-to-end before committing
5. Clear all outputs before committing: `Cell > All Output > Clear`

## License

MIT License - See [LICENSE](../LICENSE) for details
