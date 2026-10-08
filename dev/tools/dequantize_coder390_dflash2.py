"""Convert the published Coder390 per-tensor E4M3 draft to Splash BF16."""

import argparse
import hashlib
import json
import shutil
import struct
from pathlib import Path

import numpy as np


def e4m3_table():
    codes = np.arange(256, dtype=np.uint8)
    exponent = ((codes >> 3) & 15).astype(np.int32)
    fraction = (codes & 7).astype(np.float32)
    values = np.where(
        exponent == 0,
        fraction * 2**-9,
        (1 + fraction / 8) * np.exp2(exponent - 7),
    ).astype(np.float32)
    values[codes >= 128] *= -1
    values[(codes & 127) == 127] = np.nan
    return values


def bf16_bits(values):
    bits = np.asarray(values, dtype="<f4").view("<u4")
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype("<u2")


def self_test():
    table = e4m3_table()
    for code, expected in {
        0: 0.0,
        1: 2**-9,
        7: 7 * 2**-9,
        8: 2**-6,
        56: 1.0,
        57: 1.125,
        126: 448.0,
        184: -1.0,
        254: -448.0,
    }.items():
        assert float(table[code]) == expected
    assert np.signbit(table[128]) and np.isnan(table[127]) and np.isnan(table[255])
    halfway = np.array([0x3F808000, 0x3F818000, 0xBF808000, 0xBF818000], dtype="<u4")
    assert bf16_bits(halfway.view("<f4")).tolist() == [0x3F80, 0x3F82, 0xBF80, 0xBF82]


def convert(source, destination):
    self_test()
    manifest = json.loads((source / "manifest.json").read_text())
    checkpoint = source / "model.safetensors"
    with checkpoint.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    assert digest == manifest["files"]["model.safetensors"]["sha256"]
    with checkpoint.open("rb") as file:
        header_size = struct.unpack("<Q", file.read(8))[0]
        header = json.loads(file.read(header_size))
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    fp8 = {k for k, v in tensors.items() if v["dtype"] == "F8_E4M3"}
    scales = {k.removesuffix(".weight") + ".weight_scale" for k in fp8}
    assert len(fp8) == len(scales) == 20 and len(tensors) == 101
    assert all(
        tensors[k]["dtype"] == "F32" and tensors[k]["shape"] == [] for k in scales
    )
    assert all(
        v["dtype"] == "BF16" for k, v in tensors.items() if k not in fp8 | scales
    )
    data = np.memmap(checkpoint, mode="r", dtype=np.uint8, offset=8 + header_size)
    output_header, offset = {}, 0
    for name, tensor in tensors.items():
        if name in scales:
            continue
        begin, end = tensor["data_offsets"]
        size = (end - begin) * (2 if name in fp8 else 1)
        output_header[name] = {
            "dtype": "BF16",
            "shape": tensor["shape"],
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(output_header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    destination.mkdir(parents=True)
    pending = destination / "model.safetensors.partial"
    table = e4m3_table()
    with pending.open("wb") as output:
        output.write(struct.pack("<Q", len(encoded)))
        output.write(encoded)
        for name, tensor in tensors.items():
            if name in scales:
                continue
            begin, end = tensor["data_offsets"]
            scale = None
            if name in fp8:
                scale_name = name.removesuffix(".weight") + ".weight_scale"
                start, stop = tensors[scale_name]["data_offsets"]
                scale = np.frombuffer(data[start:stop], dtype="<f4")[0]
                assert np.isfinite(scale) and scale > 0
            for start in range(begin, end, 8 * 1024 * 1024):
                chunk = data[start : min(start + 8 * 1024 * 1024, end)]
                if scale is None:
                    output.write(chunk)
                else:
                    values = table[chunk] * scale
                    assert np.isfinite(values).all(), name
                    output.write(bf16_bits(values).tobytes())
            print("Converted" if scale is not None else "Preserved", name, flush=True)
    pending.rename(destination / "model.safetensors")
    config = json.loads((source / "config.json").read_text())
    config.pop("quantization_config")
    config["dtype"] = "bfloat16"
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    shutil.copyfile(source / "manifest.json", destination / "source-manifest.json")
    with (destination / "model.safetensors").open("rb") as file:
        output_digest = hashlib.file_digest(file, "sha256").hexdigest()
    (destination / "conversion.json").write_text(
        json.dumps(
            {
                "source_sha256": digest,
                "output_sha256": output_digest,
                "algorithm": "E4M3FN times FP32 per-tensor scale, BF16 round-to-nearest-even",
                "converted_tensors": 20,
                "preserved_bf16_tensors": 61,
            },
            indent=2,
        )
        + "\n"
    )
    print("DONE", destination, output_digest, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    convert(args.source, args.destination)
