"""Command-line entry point for reproducible BIDSleep preparation."""
import argparse
import os
from pathlib import Path

from .features import prepare_official_bidsleep


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare BIDSleep 30-second epochs")
    parser.add_argument("--output-dir", default="ml/artifacts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    raw = os.environ.get("SMART_SLEEP_DATA_DIR")
    if not raw:
        parser.error("SMART_SLEEP_DATA_DIR must point to raw BIDSleep files outside the repository")
    prepare_official_bidsleep(Path(raw), Path(args.output_dir), args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
