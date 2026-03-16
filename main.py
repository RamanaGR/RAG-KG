#!/usr/bin/env python3
"""
GraphRAG CLI: ask a question and get an answer using hybrid (vector + graph) retrieval.
Usage:
  python main.py                    # interactive
  python main.py "Your question?"   # single question
  python main.py --ingest path.pdf  # ingest a file first
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

from src.retrieval import query_rag
from src.ingest import ingest_file


def main():
    parser = argparse.ArgumentParser(description="GraphRAG: ask questions over your documents (Neo4j + Vector + Graph)")
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="Question to answer (optional; if omitted, run in interactive mode)",
    )
    parser.add_argument(
        "--ingest",
        metavar="FILE",
        help="Ingest a PDF or text file before running (e.g. --ingest doc.pdf)",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Exit after one question instead of prompting again",
    )
    args = parser.parse_args()

    if args.ingest:
        path = Path(args.ingest)
        if not path.exists():
            print(f"Error: file not found: {path}", file=sys.stderr)
            sys.exit(1)
        print(f"Ingesting {path}...")
        try:
            n = ingest_file(str(path))
            print(f"Done. Ingested {n} chunks.")
        except Exception as e:
            print(f"Ingest failed: {e}", file=sys.stderr)
            sys.exit(1)
        if not args.question and args.no_interactive:
            return

    def answer(q: str) -> None:
        q = q.strip()
        if not q:
            return
        try:
            response = query_rag(q)
            print(response)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)

    if args.question:
        answer(args.question)
        if not args.no_interactive:
            print("\n--- Ask another question (empty line to exit) ---")
            while True:
                try:
                    line = input("> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not line:
                    break
                answer(line)
        return

    # Interactive
    print("GraphRAG CLI. Type a question and press Enter (empty line to exit).")
    try:
        while True:
            line = input("> ").strip()
            if not line:
                break
            answer(line)
    except (EOFError, KeyboardInterrupt):
        pass
    print("Bye.")


if __name__ == "__main__":
    main()
