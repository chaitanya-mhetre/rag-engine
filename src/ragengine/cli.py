"""`rag` command-line interface for local, offline use (no API server needed)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ragengine.ingestion.chunking import RecursiveChunker, TokenWindowChunker
from ragengine.ingestion.pipeline import Ingestor


def _cmd_ingest(args: argparse.Namespace) -> int:
    chunker = (
        TokenWindowChunker(args.size, args.overlap)
        if args.chunker == "window"
        else RecursiveChunker(args.size, args.overlap)
    )
    ingestor = Ingestor(chunker=chunker)
    for path in args.paths:
        doc = ingestor.ingest(Path(path).read_bytes(), Path(path).name)
        print(f"# {doc.parsed.title} ({doc.parsed.mime_type}) -> {len(doc.chunks)} chunks")
        for c in doc.chunks:
            where = " > ".join(c.heading_path) or "-"
            page = f" p{c.page}" if c.page else ""
            preview = c.text[:80].replace("\n", " ")
            print(f"  [{c.ordinal:03d}] {c.token_count:4d} tok  {where}{page} | {preview}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="parse and chunk files, print the chunks")
    ingest.add_argument("paths", nargs="+")
    ingest.add_argument("--chunker", choices=["recursive", "window"], default="recursive")
    ingest.add_argument("--size", type=int, default=256)
    ingest.add_argument("--overlap", type=int, default=32)
    ingest.set_defaults(func=_cmd_ingest)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
