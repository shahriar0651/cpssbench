"""Small command-line helper: ``python -m cpssbench list``."""

from __future__ import annotations

import argparse

from . import describe, list_datasets, load


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and inspect vehicular IDS datasets.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List registered datasets")

    info = sub.add_parser("info", help="Show one dataset specification")
    info.add_argument("name")

    fetch = sub.add_parser("download", help="Download and preprocess a split")
    fetch.add_argument("name")
    fetch.add_argument("--root", default="./data")
    fetch.add_argument("--split", default="train", choices=["train", "test"])

    args = parser.parse_args()
    if args.command == "list":
        for row in list_datasets():
            shape = row["input_shape"] or "n/a"
            print(f"{row['name']:16} {row['status']:8} {row['family']:4} {shape}")
    elif args.command == "info":
        spec = describe(args.name)
        print(spec)
    else:
        dataset = load(args.name, root=args.root, split=args.split, download=True, verbose=True)
        print(f"Loaded {len(dataset)} windows with shape {dataset.input_shape}")


if __name__ == "__main__":
    main()
