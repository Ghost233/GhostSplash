"""Build an explicitly synthetic zero-payload fixture; read source headers only."""

import argparse
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspector", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("shards", type=Path, nargs="+")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to replace an existing fixture directory")
    info = json.loads(
        subprocess.check_output(
            [str(args.inspector), "--header-only", *map(str, args.shards)],
            text=True,
        )
    )
    target = args.output / "target"
    target.mkdir(parents=True)
    records = []
    for number, shard in enumerate(info["shards"], 1):
        name = f"SYNTHETIC-ZERO-Flash-Next-{number:05}-of-00002.gguf"
        path = target / name
        with Path(shard["path"]).open("rb") as source:
            header = source.read(shard["header_bytes"])
        if len(header) != shard["header_bytes"]:
            raise SystemExit("source header is truncated")
        with path.open("xb") as fixture:
            fixture.write(header)
            fixture.flush()
            os.ftruncate(fixture.fileno(), shard["required_file_bytes"])
        stat = path.stat()
        allocated = stat.st_blocks * 512
        if allocated > len(header) + (16 << 20):
            raise SystemExit(
                "filesystem allocated a nonsparse payload; fixture rejected"
            )
        records.append(
            {
                "path": str(path),
                "source_header_path": shard["path"],
                "source_payload_bytes_copied": 0,
                "header_bytes": len(header),
                "declared_bytes": stat.st_size,
                "physical_bytes": allocated,
            }
        )
    marker = {
        "fixture": "SYNTHETIC ZERO WEIGHTS: no actual model answer or quality claim",
        "all_payload_bytes_zero": True,
        "source_payload_bytes_copied": 0,
        "shards": records,
    }
    (args.output / "SYNTHETIC-ZERO-FIXTURE.json").write_text(
        json.dumps(marker, indent=2) + "\n"
    )
    print(json.dumps(marker, indent=2))


if __name__ == "__main__":
    main()
