"""Configuration loader for RAG demo."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv


class ConfigLoader:
    """Load and manage configuration from YAML files and environment variables."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader.

        Args:
            config_path: Path to config YAML file. If None, uses default.
        """
        # Load environment variables
        load_dotenv()

        # Determine config path
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "configs" / "config.yaml"

        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            self._config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated key.

        Args:
            key: Dot-separated key path (e.g., "embeddings.provider")
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_embeddings_config(self) -> Dict[str, Any]:
        """Get embeddings configuration."""
        return self.get("embeddings", {})

    def get_vector_db_config(self) -> Dict[str, Any]:
        """Get vector database configuration."""
        return self.get("vector_db", {})

    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration."""
        return self.get("llm", {})

    def get_chunking_config(self) -> Dict[str, Any]:
        """Get chunking configuration."""
        return self.get("chunking", {})

    def get_retrieval_config(self) -> Dict[str, Any]:
        """Get retrieval configuration."""
        return self.get("retrieval", {})

    def get_sources_config(self) -> Dict[str, Any]:
        """Get documentation sources configuration."""
        return self.get("sources", {})

    def get_rag_config(self) -> Dict[str, Any]:
        """Get RAG pipeline configuration."""
        return self.get("rag", {})

    @property
    def config(self) -> Dict[str, Any]:
        """Get full configuration dictionary."""
        return self._config
