"""Command-line entry points; native dependencies are loaded only when requested."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Anchored Planning reproduction tools")
    sub = parser.add_subparsers(dest="command", required=True)
    analyze = sub.add_parser("analyze", help="Summarize simulator outcomes on CPU")
    analyze.add_argument("--study", choices=("main", "local-target"), required=True)
    analyze.add_argument("--root", type=Path, default=Path.cwd(), help="Folder containing protocols/")
    analyze.add_argument("--outcomes", type=Path, required=True, help="Directory of outcomes_*.jsonl files")
    analyze.add_argument("--output", type=Path, required=True)
    for command, description in (("inspect-assets", "Inspect external file schemas and dependency versions"),
                                  ("encode-cache", "Encode an external dataset into a retrieval cache"),
                                  ("evaluate", "Run frozen queries in the native simulator")):
        p = sub.add_parser(command, help=description)
        p.add_argument("--assets", type=Path, required=True, help="Asset configuration JSON; paths are relative to this file")
        p.add_argument("--task", choices=("cube", "pusht", "reacher", "tworoom"), required=True)
        if command != "inspect-assets":
            p.add_argument("--device", default="cuda:0")
        if command == "encode-cache":
            p.add_argument("--batch-size", type=int, default=128)
        if command == "evaluate":
            p.add_argument("--root", type=Path, default=Path.cwd())
            p.add_argument("--study", choices=("main", "local-target"), required=True)
            p.add_argument("--output", type=Path, required=True)
            p.add_argument("--limit", type=int, help="First N queries; omitted for the full frozen cohort")
            p.add_argument("--shards", type=int, default=1)
            p.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    try:
        if args.command == "analyze":
            from .analysis import run
        else:
            from .runtime import run
        run(args)
    except (FileNotFoundError, ValueError, ModuleNotFoundError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
