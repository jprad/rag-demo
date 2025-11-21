#!/usr/bin/env python3
"""Script to test RAG pipeline with queries."""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline.rag import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Query the RAG pipeline")
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        default=None,
        help="Query to ask",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of context documents to retrieve",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Filter by source (e.g., 'ansible', 'terraform')",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("RAG Demo - Query Interface")
    print("=" * 60)

    # Initialize pipeline
    print("\nInitializing RAG pipeline...")
    pipeline = RAGPipeline(config_path=args.config)
    print("Pipeline ready!")

    # Build filters
    filters = None
    if args.source:
        filters = {"source_name": args.source}
        print(f"Filtering by source: {args.source}")

    if args.interactive or not args.query:
        # Interactive mode
        print("\nInteractive mode. Type 'exit' or 'quit' to exit, 'clear' to clear history.\n")

        while True:
            try:
                query = input("\nYou: ").strip()

                if query.lower() in ["exit", "quit"]:
                    print("Goodbye!")
                    break

                if query.lower() == "clear":
                    pipeline.clear_history()
                    print("Conversation history cleared.")
                    continue

                if not query:
                    continue

                print("\nAssistant: ", end="", flush=True)

                # Stream response
                for chunk in pipeline.query_stream(
                    question=query,
                    top_k=args.top_k,
                    filters=filters,
                ):
                    print(chunk, end="", flush=True)

                print("\n")

                # Get sources
                response = pipeline.query(
                    question=query,
                    top_k=args.top_k,
                    filters=filters,
                )

                if response.get("sources"):
                    print("\nSources:")
                    for i, source in enumerate(response["sources"], 1):
                        print(f"  {i}. [{source['metadata'].get('source_name', 'unknown')}] "
                              f"(score: {source['score']:.3f})")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")

    else:
        # Single query mode
        print(f"\nQuery: {args.query}\n")

        try:
            response = pipeline.query(
                question=args.query,
                top_k=args.top_k,
                filters=filters,
            )

            print("Answer:")
            print("-" * 60)
            print(response["answer"])
            print("-" * 60)

            if response.get("sources"):
                print("\nSources:")
                for i, source in enumerate(response["sources"], 1):
                    print(f"\n[Source {i}] - {source['metadata'].get('source_name', 'unknown')} "
                          f"(score: {source['score']:.3f})")
                    print(source["text"])
                    print("-" * 60)

        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
