import contextlib
import io
import json
import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType, GGUFWriter

from dev.tools.dequantize_dflash2_gguf import (
    convert,
    converted_chunks,
    draft_config,
    tensor_mapping,
)
from install.gguf import Metadata


def quantized_block(kind):
    raw = np.zeros(GGML_QUANT_SIZES[kind][1], dtype=np.uint8)
    if kind == GGMLQuantizationType.Q4_K:
        raw[:2] = np.frombuffer(struct.pack("<e", 0.25), dtype=np.uint8)
        raw[2:4] = np.frombuffer(struct.pack("<e", 0.125), dtype=np.uint8)
        raw[4:8], raw[8:12], raw[12:16], raw[16:] = 3, 1, 0x13, 0x21
    elif kind == GGMLQuantizationType.Q6_K:
        raw[:128], raw[128:192], raw[192:208] = 0x98, 0xAA, 3
        raw[208:] = np.frombuffer(struct.pack("<e", 0.125), dtype=np.uint8)
    elif kind == GGMLQuantizationType.Q8_0:
        raw[:2] = np.frombuffer(struct.pack("<e", 0.125), dtype=np.uint8)
        raw[2:] = np.arange(-16, 16, dtype=np.int8).view(np.uint8)
    return raw


class DraftGgufConversionTests(unittest.TestCase):
    def fixture(
        self,
        source,
        target,
        *,
        architecture="qwen35",
        fc_kind=GGMLQuantizationType.Q4_K,
    ):
        values = {
            "general.architecture": "dflash",
            "dflash.block_count": 2,
            "dflash.embedding_length": 256,
            "dflash.feed_forward_length": 512,
            "dflash.attention.head_count": 4,
            "dflash.attention.head_count_kv": 2,
            "dflash.attention.key_length": 64,
            "dflash.attention.sliding_window": 2048,
            "dflash.context_length": 262144,
            "dflash.attention.causal": False,
            "dflash.attention.sliding_window_pattern": [True, True],
            "dflash.target_layers": [1, 3],
            "dflash.rope.freq_base": 10000000.0,
            "dflash.attention.layer_norm_rms_epsilon": 1e-6,
            "dflash.block_size": 8,
            "dflash.conv_group_size": 16,
            "dflash.conv_kernel_size": 2,
            "dflash.selector_rank": 256,
            "dflash.selector_top_k": 16,
            "tokenizer.ggml.mask_token_id": 250,
            "tokenizer.ggml.tokens": [f"token{i}" for i in range(256)],
            "tokenizer.ggml.token_type": [1] * 256,
            "tokenizer.ggml.merges": ["a b"],
        }

        def add_metadata(writer, data):
            for key, value in data.items():
                if key == "general.architecture":
                    continue
                if isinstance(value, list):
                    writer.add_array(key, value)
                elif isinstance(value, bool):
                    writer.add_bool(key, value)
                elif isinstance(value, int):
                    writer.add_uint32(key, value)
                elif isinstance(value, float):
                    writer.add_float32(key, value)
                else:
                    writer.add_string(key, value)

        def finish(writer):
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()

        writer = GGUFWriter(str(target), architecture)
        add_metadata(
            writer,
            {
                architecture + ".block_count": 4,
                architecture + ".embedding_length": 256,
                **{k: v for k, v in values.items() if k.startswith("tokenizer.")},
            },
        )
        finish(writer)
        writer = GGUFWriter(str(source), "dflash")
        add_metadata(writer, values)
        config = {
            "hidden_size": 256,
            "intermediate_size": 512,
            "head_dim": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 256,
            "dflash_config": {
                "selector_rank": 256,
                "target_layer_ids": [0, 2],
                "conv_kernel_size": 2,
                "conv_group_size": 16,
            },
        }
        for name, (_, shape) in tensor_mapping(config).items():
            if name in ("fc.weight", "blk.1.ffn_down.weight"):
                kind = fc_kind if name == "fc.weight" else GGMLQuantizationType.Q6_K
                blocks = np.prod(shape) // GGML_QUANT_SIZES[kind][0]
                data = np.tile(quantized_block(kind), int(blocks)).reshape(shape[0], -1)
                writer.add_tensor(name, data, raw_dtype=kind)
            else:
                writer.add_tensor(name, np.full(shape, 1.25, dtype=np.float32))
        finish(writer)

    def test_q4k_scale_offset_and_q6k_signed_codes(self):
        for kind, expected in (
            (GGMLQuantizationType.Q4_K, {0x3F20, 0x3FB0}),
            (GGMLQuantizationType.Q6_K, {0x4040, 0x4058}),
        ):
            with self.subTest(kind=kind):
                tensor = SimpleNamespace(
                    name="weight", tensor_type=kind, data=quantized_block(kind)
                )
                bits = np.concatenate(list(converted_chunks(tensor)))
                self.assertEqual(len(bits), 256)
                self.assertEqual(set(map(int, bits)), expected)

    def test_nonfinite_and_unsupported_types_are_rejected(self):
        tensor = SimpleNamespace(
            name="weight",
            tensor_type=GGMLQuantizationType.F32,
            data=np.array([np.nan], dtype=np.float32),
        )
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            list(converted_chunks(tensor))
        tensor.tensor_type = GGMLQuantizationType.Q5_K
        with self.assertRaisesRegex(ValueError, "unsupported"):
            list(converted_chunks(tensor))

    def test_q8_0_signed_values_and_half_scale(self):
        tensor = SimpleNamespace(
            name="weight",
            tensor_type=GGMLQuantizationType.Q8_0,
            data=quantized_block(GGMLQuantizationType.Q8_0),
        )
        actual = np.concatenate(list(converted_chunks(tensor)))
        expected = (np.arange(-16, 16, dtype=np.float32) * 0.125).view(np.uint32) >> 16
        np.testing.assert_array_equal(actual, expected)

    def test_flash_next_q8_conversion_preserves_shapes_and_capture_mapping(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, target, destination = (
                root / "draft.gguf",
                root / "target.gguf",
                root / "output",
            )
            self.fixture(
                source,
                target,
                architecture="qwen4exp",
                fc_kind=GGMLQuantizationType.Q8_0,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                convert(source, destination, target)
            config = json.loads((destination / "config.json").read_text())
            self.assertEqual(config["dflash_config"]["target_layer_ids"], [0, 2])
            self.assertEqual(config["num_target_layers"], 4)
            self.assertEqual(config["hidden_size"], 256)
            payload = (destination / "model.safetensors").read_bytes()
            size = struct.unpack_from("<Q", payload)[0]
            header = json.loads(payload[8 : 8 + size])
            self.assertEqual(header["fc.weight"]["shape"], [256, 512])
            begin, end = header["fc.weight"]["data_offsets"]
            actual = np.frombuffer(
                payload[8 + size + begin : 8 + size + end], dtype="<u2"
            )
            expected = np.tile(
                (np.arange(-16, 16, dtype=np.float32) * 0.125).view(np.uint32) >> 16,
                256 * 512 // 32,
            )
            np.testing.assert_array_equal(actual, expected)

    def test_complete_conversion_shapes_layer_ids_and_no_overwrite(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, target, destination = (
                root / "draft.gguf",
                root / "target.gguf",
                root / "output",
            )
            self.fixture(source, target)
            with contextlib.redirect_stdout(io.StringIO()):
                convert(source, destination, target)
            config = json.loads((destination / "config.json").read_text())
            self.assertEqual(config["dflash_config"]["target_layer_ids"], [0, 2])
            self.assertEqual(config["architectures"], ["DFlash2DraftModel"])
            self.assertEqual(config["rms_norm_eps"], 1e-6)
            payload = (destination / "model.safetensors").read_bytes()
            size = struct.unpack_from("<Q", payload)[0]
            tensors = json.loads(payload[8 : 8 + size])
            self.assertEqual(len(tensors), 36)
            self.assertEqual(tensors["fc.weight"]["shape"], [256, 512])
            self.assertEqual(
                tensors["layers.1.attention_conv.base_kernel"]["shape"], [2, 2, 256]
            )
            self.assertEqual(
                tensors["candidate_selector.predecessor_codebook"]["shape"], [256, 256]
            )
            for name, tensor in tensors.items():
                begin, end = tensor["data_offsets"]
                bits = np.frombuffer(
                    payload[8 + size + begin : 8 + size + end], dtype="<u2"
                )
                self.assertEqual(tensor["dtype"], "BF16")
                expected = (
                    {0x3F20, 0x3FB0}
                    if name == "fc.weight"
                    else (
                        {0x4040, 0x4058}
                        if name == "layers.1.mlp.down_proj.weight"
                        else {0x3FA0}
                    )
                )
                self.assertEqual(set(map(int, bits)), expected)
            with self.assertRaises(FileExistsError):
                convert(source, destination, target)

    def test_tokenizer_mismatch_is_rejected(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root / "draft.gguf", root / "target.gguf")
            source, target = (
                Metadata(root / "draft.gguf"),
                Metadata(root / "target.gguf"),
            )
            target.values["tokenizer.ggml.tokens"][0] = "other"
            with self.assertRaisesRegex(ValueError, "tokenizer differ"):
                draft_config(source, target)
