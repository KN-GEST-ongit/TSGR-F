"""Import one or more completed SGRF evaluation shards idempotently."""

from __future__ import annotations

import argparse
from pathlib import Path

from tsgr.baselines.sgrf_distributed import import_sgrf_evaluation_shard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard_zip", type=Path, nargs="+")
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    args = parser.parse_args()
    imported = 0
    skipped = 0
    for shard in args.shard_zip:
        result = import_sgrf_evaluation_shard(shard, evaluation_dir=args.evaluation_dir)
        imported += int(result["imported_job_count"])
        skipped += int(result["skipped_job_count"])
        print(
            f"Imported {shard.name}: new={result['imported_job_count']} "
            f"already_present={result['skipped_job_count']}"
        )
    print(f"SGRF shard import complete: new={imported}, already_present={skipped}")


if __name__ == "__main__":
    main()
