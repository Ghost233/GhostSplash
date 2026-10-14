"""Convert a local DFlash2 GGUF to Splash's BF16 safetensors draft format.

Uses llama.cpp gguf-py's dequantization, bounded CPU chunks, and BF16 RNE.
GGUF target_layers are HF decoder indices + 1; Q/K rows need no permutation.
"""

import argparse
import hashlib
import json
import math
import struct
from importlib.metadata import version
from pathlib import Path

import numpy as np
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType, GGUFReader, dequantize

from install.gguf import Metadata

from .dequantize_coder390_dflash2 import bf16_bits

SUPPORTED = {
    GGMLQuantizationType.F32,
    GGMLQuantizationType.Q4_K,
    GGMLQuantizationType.Q6_K,
}


def draft_config(source, target):
    if source.require("general.architecture", str) != "dflash":
        raise ValueError("source is not a DFlash GGUF")
    if target.require("general.architecture", str) != "qwen35":
        raise ValueError("target must be a Qwen3.5/3.8 dense GGUF")
    for key in ("tokens", "token_type", "merges"):
        if (
            source.values["tokenizer.ggml." + key]
            != target.values["tokenizer.ggml." + key]
        ):
            raise ValueError("draft and target tokenizer differ: " + key)
    keys = {
        "num_hidden_layers": "block_count",
        "hidden_size": "embedding_length",
        "intermediate_size": "feed_forward_length",
        "num_attention_heads": "attention.head_count",
        "num_key_value_heads": "attention.head_count_kv",
        "head_dim": "attention.key_length",
        "sliding_window": "attention.sliding_window",
        "max_position_embeddings": "context_length",
    }
    config = {name: source.positive("dflash." + key) for name, key in keys.items()}
    captures = [
        int(layer) - 1 for layer in source.require("dflash.target_layers", list)
    ]
    native_layers = target.positive("qwen35.block_count") - target.values.get(
        "qwen35.nextn_predict_layers", 0
    )
    if (
        not captures
        or len(set(captures)) != len(captures)
        or any(not 0 <= i < native_layers for i in captures)
    ):
        raise ValueError("invalid draft target-layer indices")
    if config["hidden_size"] != target.positive("qwen35.embedding_length"):
        raise ValueError("draft hidden size differs from target")
    if (
        source.require("dflash.attention.causal", bool)
        or source.require("dflash.attention.sliding_window_pattern", list)
        != [True] * config["num_hidden_layers"]
    ):
        raise ValueError("only noncausal DFlash2 with all sliding layers is supported")
    flash = {
        name: source.positive("dflash." + name)
        for name in (
            "block_size",
            "conv_group_size",
            "conv_kernel_size",
            "selector_rank",
            "selector_top_k",
        )
    }
    flash.update(
        target_layer_ids=captures,
        mask_token_id=source.positive("tokenizer.ggml.mask_token_id"),
    )
    if (
        flash["block_size"],
        flash["conv_group_size"],
        flash["conv_kernel_size"],
        flash["selector_rank"],
        flash["selector_top_k"],
    ) != (8, 16, 2, 256, 16):
        raise ValueError("DFlash2 block/conv/selector configuration is unsupported")
    epsilon = source.require("dflash.attention.layer_norm_rms_epsilon", float)
    if struct.pack("<f", epsilon) != struct.pack("<f", 1e-6):
        raise ValueError("draft RMS epsilon differs from the runtime's FP32 value")
    config.update(
        architectures=["DFlash2DraftModel"],
        model_type="qwen3",
        dtype="bfloat16",
        vocab_size=len(source.values["tokenizer.ggml.tokens"]),
        num_target_layers=native_layers,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        attention_bias=False,
        is_causal=False,
        tie_word_embeddings=False,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": source.require("dflash.rope.freq_base", float),
        },
        layer_types=["sliding_attention"] * config["num_hidden_layers"],
        dflash_config=flash,
    )
    return config


def tensor_mapping(config):
    hidden, intermediate, dimension = (
        config["hidden_size"],
        config["intermediate_size"],
        config["head_dim"],
    )
    flash = config["dflash_config"]
    dynamic = 2 * flash["conv_kernel_size"] * (hidden // flash["conv_group_size"])
    mapping = {
        "fc.weight": ("fc.weight", [hidden, len(flash["target_layer_ids"]) * hidden]),
        "enc.output_norm.weight": ("hidden_norm.weight", [hidden]),
        "output_norm.weight": ("norm.weight", [hidden]),
        "selector_hidden.weight": (
            "candidate_selector.hidden_projection.weight",
            [flash["selector_rank"], hidden],
        ),
        "selector_predecessor.weight": (
            "candidate_selector.predecessor_codebook",
            [config["vocab_size"], flash["selector_rank"]],
        ),
        "selector_successor.weight": (
            "candidate_selector.successor_codebook",
            [config["vocab_size"], flash["selector_rank"]],
        ),
    }
    fields = {
        "attn_norm.weight": ("input_layernorm.weight", [hidden]),
        "attn_q.weight": (
            "self_attn.q_proj.weight",
            [config["num_attention_heads"] * dimension, hidden],
        ),
        "attn_k.weight": (
            "self_attn.k_proj.weight",
            [config["num_key_value_heads"] * dimension, hidden],
        ),
        "attn_v.weight": (
            "self_attn.v_proj.weight",
            [config["num_key_value_heads"] * dimension, hidden],
        ),
        "attn_output.weight": (
            "self_attn.o_proj.weight",
            [hidden, config["num_attention_heads"] * dimension],
        ),
        "attn_q_norm.weight": ("self_attn.q_norm.weight", [dimension]),
        "attn_k_norm.weight": ("self_attn.k_norm.weight", [dimension]),
        "ffn_norm.weight": ("post_attention_layernorm.weight", [hidden]),
        "ffn_gate.weight": ("mlp.gate_proj.weight", [intermediate, hidden]),
        "ffn_up.weight": ("mlp.up_proj.weight", [intermediate, hidden]),
        "ffn_down.weight": ("mlp.down_proj.weight", [hidden, intermediate]),
        "attn_conv_base": (
            "attention_conv.base_kernel",
            [2, flash["conv_kernel_size"], hidden],
        ),
        "attn_conv_proj.weight": (
            "attention_conv.kernel_projection.weight",
            [dynamic, hidden],
        ),
        "ffn_conv_base": (
            "mlp_conv.base_kernel",
            [2, flash["conv_kernel_size"], hidden],
        ),
        "ffn_conv_proj.weight": (
            "mlp_conv.kernel_projection.weight",
            [dynamic, hidden],
        ),
    }
    for layer in range(config["num_hidden_layers"]):
        mapping.update(
            {
                f"blk.{layer}.{src}": (f"layers.{layer}.{dst}", shape)
                for src, (dst, shape) in fields.items()
            }
        )
    return mapping


def converted_chunks(tensor):
    if tensor.tensor_type not in SUPPORTED:
        raise ValueError("unsupported GGUF tensor type: " + tensor.tensor_type.name)
    _, block_bytes = GGML_QUANT_SIZES[tensor.tensor_type]
    raw = tensor.data.view(np.uint8).reshape(-1)
    chunk_bytes = max(1, (4 * 1024 * 1024) // block_bytes) * block_bytes
    for start in range(0, len(raw), chunk_bytes):
        values = dequantize(
            raw[start : start + chunk_bytes], tensor.tensor_type
        ).reshape(-1)
        if not np.isfinite(values).all():
            raise ValueError("nonfinite source tensor: " + tensor.name)
        yield bf16_bits(values)


def convert(source, destination, target):
    source, destination, target = Path(source), Path(destination), Path(target)
    before = source.stat()
    metadata = Metadata(source)
    config = draft_config(metadata, Metadata(target))
    mapping = tensor_mapping(config)
    reader = GGUFReader(str(source))
    if {t.name for t in reader.tensors} != set(mapping):
        raise ValueError(
            "GGUF tensor names do not match the complete DFlash2 checkpoint"
        )
    header, offset = {}, 0
    for tensor in reader.tensors:
        name, shape = mapping[tensor.name]
        if (
            list(reversed(tensor.shape.tolist())) != shape
            or tensor.tensor_type not in SUPPORTED
        ):
            raise ValueError("GGUF tensor shape/type mismatch: " + tensor.name)
        size = math.prod(shape) * 2
        header[name] = {
            "dtype": "BF16",
            "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        offset += size
    with source.open("rb") as stream:
        source_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    destination.mkdir(parents=True, exist_ok=False)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    digest = hashlib.sha256()
    pending = destination / "model.safetensors.partial"
    report = []
    with pending.open("xb") as output:
        prefix = struct.pack("<Q", len(encoded)) + encoded
        output.write(prefix)
        digest.update(prefix)
        for tensor in reader.tensors:
            count = 0
            for bits in converted_chunks(tensor):
                data = bits.tobytes()
                output.write(data)
                digest.update(data)
                count += bits.size
            if count != tensor.n_elements:
                raise ValueError("converted tensor size mismatch: " + tensor.name)
            report.append(
                {
                    "source": tensor.name,
                    "destination": mapping[tensor.name][0],
                    "source_type": tensor.tensor_type.name,
                    "elements": count,
                }
            )
            print("Converted", tensor.name, flush=True)
    if (source.stat().st_size, source.stat().st_mtime_ns) != (
        before.st_size,
        before.st_mtime_ns,
    ):
        raise ValueError("source GGUF changed during conversion")
    pending.rename(destination / "model.safetensors")
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (destination / "conversion.json").write_text(
        json.dumps(
            {
                "source_gguf": str(source.resolve()),
                "source_sha256": source_digest,
                "target_gguf": str(target.resolve()),
                "target_bytes": target.stat().st_size,
                "algorithm": "llama.cpp gguf-py F32/Q4_K/Q6_K dequantization; BF16 round-to-nearest-even; GGUF shape axes reversed; target layer IDs minus one",
                "gguf_version": version("gguf"),
                "output_sha256": digest.hexdigest(),
                "output_bytes": pending.with_name("model.safetensors").stat().st_size,
                "tensors": report,
            },
            indent=2,
        )
        + "\n"
    )
    print("DONE", destination, digest.hexdigest(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    convert(args.source, args.destination, args.target)
