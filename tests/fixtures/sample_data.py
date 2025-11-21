"""Sample test data and fixtures."""

# Sample documents for testing
SAMPLE_DOCUMENTS = [
    {
        "content": """
        Ansible is an open-source automation tool for configuration management,
        application deployment, and task automation. It uses YAML syntax for
        playbooks and is agentless, connecting via SSH.
        """,
        "metadata": {
            "source": "ansible",
            "url": "https://docs.ansible.com/",
            "title": "Ansible Overview"
        }
    },
    {
        "content": """
        Terraform is an infrastructure as code tool that lets you build, change,
        and version infrastructure safely and efficiently. It uses HCL
        (HashiCorp Configuration Language) for defining infrastructure.
        """,
        "metadata": {
            "source": "terraform",
            "url": "https://terraform.io/docs/",
            "title": "Terraform Introduction"
        }
    },
    {
        "content": """
        Packer is a tool for creating identical machine images for multiple
        platforms from a single source configuration. It supports various
        platforms including AWS, Azure, GCP, and VMware.
        """,
        "metadata": {
            "source": "packer",
            "url": "https://packer.io/docs/",
            "title": "Packer Overview"
        }
    }
]

# Sample queries for testing
SAMPLE_QUERIES = [
    "How do I create an Ansible playbook?",
    "What is Terraform used for?",
    "How do I build a VM image with Packer?",
    "What is the difference between Ansible and Terraform?",
    "How do I configure multiple machines with Vagrant?"
]

# Sample embeddings (simplified for testing)
SAMPLE_EMBEDDINGS = [
    [0.1, 0.2, 0.3, 0.4, 0.5],
    [0.2, 0.3, 0.4, 0.5, 0.6],
    [0.3, 0.4, 0.5, 0.6, 0.7]
]

# Sample configuration
SAMPLE_CONFIG = {
    "embeddings": {
        "provider": "sentence-transformers",
        "config": {
            "model_name": "all-MiniLM-L6-v2",
            "device": "cpu"
        }
    },
    "vector_db": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333,
            "collection_name": "test_docs"
        }
    },
    "llm": {
        "provider": "ollama",
        "config": {
            "base_url": "http://localhost:11434",
            "model": "llama3",
            "temperature": 0.7,
            "max_tokens": 512
        }
    },
    "chunking": {
        "strategy": "recursive",
        "config": {
            "chunk_size": 1000,
            "chunk_overlap": 200
        }
    },
    "retrieval": {
        "strategy": "semantic",
        "config": {
            "top_k": 5,
            "score_threshold": 0.7
        }
    }
}
