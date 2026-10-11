"""Write a model-bound compact draft ranking; does not infer token popularity."""

import argparse
import json
import struct
from pathlib import Path


def write_ranking(path, target_manifest, token_order):
    if len(target_manifest) != 64 or any(
        character not in "0123456789abcdef" for character in target_manifest
    ):
        raise ValueError("target manifest must be a lowercase SHA-256")
    vocabulary = len(token_order)
    if not vocabulary or any(type(token) is not int for token in token_order):
        raise ValueError("token order must contain integer token IDs")
    if sorted(token_order) != list(range(vocabulary)):
        raise ValueError("token order must be the complete vocabulary permutation")
    body = b"SPLCHD01" + target_manifest.encode("ascii")
    body += struct.pack("<I", vocabulary)
    body += struct.pack(f"<{vocabulary}I", *token_order)
    with Path(path).open("xb") as stream:
        stream.write(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-manifest", required=True)
    parser.add_argument("--token-order", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_ranking(
        args.output,
        args.target_manifest,
        json.loads(args.token_order.read_text(encoding="utf-8")),
    )


if __name__ == "__main__":
    main()
