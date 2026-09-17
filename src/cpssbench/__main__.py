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
    fetch.add_argument(
        "--filling",
        default=None,
        choices=["forward", "none", "nan", "zero", "ffill", "0"],
        help="CAN only: how to treat intermittent bus gaps (forward / none|nan / zero).",
    )

    args = parser.parse_args()
    if args.command == "list":
        for row in list_datasets():
            shape = row["input_shape"] or "n/a"
            print(f"{row['name']:16} {row['status']:8} {row['family']:4} {shape}")
    elif args.command == "info":
        spec = describe(args.name)
        print(spec)
    else:
        kwargs = dict(root=args.root, split=args.split, download=True, verbose=True)
        if args.filling is not None:
            kwargs["filling"] = args.filling
        dataset = load(args.name, **kwargs)
        suffix = f" filling={dataset.filling}" if describe(args.name).family == "can" else ""
        print(f"Loaded {len(dataset)} windows with shape {dataset.input_shape}{suffix}")


if __name__ == "__main__":
    main()
