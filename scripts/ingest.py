#!/usr/bin/env python3
"""Script to ingest documentation into vector database."""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline.ingestion import IngestionPipeline


def main():
    parser = argparse.ArgumentParser(description="Ingest documentation into vector database")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Specific source to ingest (e.g., 'ansible', 'terraform')",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate collection (delete existing data)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("RAG Demo - Document Ingestion")
    print("=" * 60)

    # Initialize pipeline
    pipeline = IngestionPipeline(config_path=args.config)

    # Recreate collection if requested
    if args.recreate:
        collection_name = pipeline.collection_name
        if pipeline.vector_store.collection_exists(collection_name):
            print(f"\nDeleting existing collection: {collection_name}")
            pipeline.vector_store.delete_collection(collection_name)
        pipeline._ensure_collection()

    # Ingest documents
    if args.source:
        # Ingest specific source
        sources_config = pipeline.config.get_sources_config()
        if args.source not in sources_config:
            print(f"Error: Unknown source '{args.source}'")
            print(f"Available sources: {', '.join(sources_config.keys())}")
            sys.exit(1)

        source_config = sources_config[args.source]
        stats = pipeline.ingest_source(
            source=source_config["url"],
            source_name=args.source,
            max_depth=source_config.get("max_depth", 1),
            include_patterns=source_config.get("include_patterns"),
            exclude_patterns=source_config.get("exclude_patterns"),
        )

        print("\n" + "=" * 60)
        print("Ingestion Complete!")
        print("=" * 60)
        print(f"Source: {stats['source']}")
        print(f"Documents: {stats['documents']}")
        print(f"Chunks: {stats['chunks']}")

    else:
        # Ingest all sources
        all_stats = pipeline.ingest_all_sources()

        print("\n" + "=" * 60)
        print("Ingestion Complete!")
        print("=" * 60)

        total_docs = sum(s.get("documents", 0) for s in all_stats)
        total_chunks = sum(s.get("chunks", 0) for s in all_stats)

        print(f"\nTotal documents: {total_docs}")
        print(f"Total chunks: {total_chunks}")

        print("\nPer-source statistics:")
        for stats in all_stats:
            if "error" in stats:
                print(f"  - {stats['source']}: ERROR - {stats['error']}")
            else:
                print(f"  - {stats['source']}: {stats['documents']} docs, {stats['chunks']} chunks")

    # Show collection info
    print("\nCollection information:")
    info = pipeline.get_stats()
    for key, value in info.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
