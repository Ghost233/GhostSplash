"""Write public compact rankings from a token order or local UTF-8 corpus.

Corpus generation uses the existing tokenizer on CPU; it never starts inference.
Reported coverage is corpus-wide, not a guarantee for every request.
"""

import argparse
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path


def write_ranking(path, token_order):
    vocabulary = len(token_order)
    if not vocabulary or any(type(token) is not int for token in token_order):
        raise ValueError("token order must contain integer token IDs")
    if sorted(token_order) != list(range(vocabulary)):
        raise ValueError("token order must be the complete vocabulary permutation")
    body = b"SPLCHD02" + struct.pack("<I", vocabulary)
    body += struct.pack(f"<{vocabulary}I", *token_order)
    with Path(path).open("xb") as stream:
        stream.write(body)


def corpus_counts(tokenizer, paths):
    counts = Counter()
    for path in paths:
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                counts.update(tokenizer.encode(line, add_special_tokens=False).ids)
    return counts


def rank_corpus(tokenizer, paths):
    counts = corpus_counts(tokenizer, paths)
    if not counts:
        raise ValueError("corpus contains no tokens")
    special = {
        token_id
        for token_id, token in tokenizer.get_added_tokens_decoder().items()
        if token.special
    }
    order = sorted(
        range(tokenizer.get_vocab_size()),
        key=lambda token_id: (token_id not in special, -counts[token_id], token_id),
    )
    return order, counts


def corpus_coverage(counts, order, fixed_rows):
    total = sum(counts.values())
    included = set(order[:fixed_rows])
    covered = sum(count for token_id, count in counts.items() if token_id in included)
    return {
        "tokens": total,
        "observed_token_ids": len(counts),
        "fixed_rows": fixed_rows,
        "token_coverage_percent": 100 * covered / total if total else None,
        "outside_fixed_tokens": total - covered,
        "outside_fixed_token_ids": sum(token_id not in included for token_id in counts),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--token-order", type=Path)
    source.add_argument("--corpus", type=Path, nargs="+")
    source.add_argument(
        "--import-legacy", type=Path, help="one-time SPLCHD01 conversion"
    )
    parser.add_argument(
        "--tokenizer", type=Path, help="tokenizer.json or its directory"
    )
    parser.add_argument("--validation-corpus", type=Path, nargs="+")
    parser.add_argument(
        "--tokenizer-profile", help="informational profile, not a binding"
    )
    parser.add_argument("--corpus-source", help="informational corpus provenance")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "schema_version": 1,
        "format_version": 2,
        "tokenizer_profile": args.tokenizer_profile,
        "corpus_source": args.corpus_source,
        "output": str(args.output.resolve()),
    }
    if not args.corpus:
        if args.tokenizer or args.validation_corpus:
            parser.error("--tokenizer and --validation-corpus require --corpus")
        if args.import_legacy:
            raw = args.import_legacy.read_bytes()
            if len(raw) < 76 or raw[:8] != b"SPLCHD01":
                raise ValueError("invalid legacy compact ranking header")
            vocabulary = struct.unpack_from("<I", raw, 72)[0]
            if len(raw) != 76 + 4 * vocabulary:
                raise ValueError("invalid legacy compact ranking size")
            order = list(struct.unpack_from(f"<{vocabulary}I", raw, 76))
            report.update(
                source="legacy_import",
                imported_from=str(args.import_legacy.resolve()),
                original_file_sha256=hashlib.sha256(raw).hexdigest(),
            )
        else:
            order = json.loads(args.token_order.read_text(encoding="utf-8"))
            report["source"] = "token_order"
    else:
        if not args.tokenizer:
            parser.error("--corpus requires --tokenizer")
        from tokenizers import Tokenizer

        path = args.tokenizer
        if path.is_dir():
            path = path / "tokenizer.json"
        codec = Tokenizer.from_file(str(path))
        order, counts = rank_corpus(codec, args.corpus)
        validation = corpus_counts(codec, args.validation_corpus or [])
        capacity = min(98304, len(order) // 2 // 256 * 256)
        fixed_rows = max(0, capacity - 256)
        report.update(
            source="corpus_frequency",
            tokenizer_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            special_tokens_reserved=sum(
                token.special for token in codec.get_added_tokens_decoder().values()
            ),
            corpus_files=[str(p.resolve()) for p in args.corpus],
            validation_files=[str(p.resolve()) for p in args.validation_corpus or []],
            training=corpus_coverage(counts, order, fixed_rows),
            validation=corpus_coverage(validation, order, fixed_rows),
            coverage_scope="aggregate token frequency; per-request fallback rules unchanged",
        )
    write_ranking(args.output, order)
    report["vocabulary_size"] = len(order)
    report["output_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
