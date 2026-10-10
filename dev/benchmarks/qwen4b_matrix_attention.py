"""Generate the matrix-only attention object without changing server policy."""

import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text()
old = "kernel == KernelLayout::Kv4Group4 && layout.format == kv::Format::Int8\n      ? 1U : prefillSplits(tiles);"
if source.count(old) != 1:
    raise SystemExit("production tail split policy changed; review matrix override")
source = source.replace(
    old,
    "kernel == KernelLayout::Kv4Group4\n      ? matrixTailSplits() : prefillSplits(tiles);",
)
anchor = "namespace splash::ops {"
if source.count(anchor) != 1:
    raise SystemExit("attention namespace anchor changed")
source = source.replace(
    anchor,
    anchor
    + "\nuint32_t matrixTailSplits();\nvoid matrixRecordTailSplit(uint32_t rows, uint32_t splits);",
    1,
)
encode_anchor = "  const kv::PrefillAttentionParams params{"
if source.count(encode_anchor) != 1:
    raise SystemExit("attention encode anchor changed")
source = source.replace(
    encode_anchor,
    "  matrixRecordTailSplit(plan.rows, plan.splits);\n" + encode_anchor,
    1,
)
Path(sys.argv[2]).write_text(source)
