import contextlib
import hashlib
import io
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dev.tools.dequantize_coder390_dflash2 import bf16_bits, convert, e4m3_table


class Coder390DraftConversionTests(unittest.TestCase):
    def test_all_e4m3_encodings(self):
        actual = e4m3_table()
        for code in range(256):
            exponent, fraction = (code & 127) >> 3, code & 7
            if exponent == 15 and fraction == 7:
                self.assertTrue(np.isnan(actual[code]))
                continue
            magnitude = (
                math.ldexp(fraction, -9)
                if exponent == 0
                else math.ldexp(8 + fraction, exponent - 10)
            )
            expected = -magnitude if code & 128 else magnitude
            self.assertEqual(
                struct.pack("<f", actual[code]), struct.pack("<f", expected)
            )

    def test_bf16_rounds_halfway_to_even(self):
        source = np.array(
            [1 + 2**-8, 1 + 3 * 2**-8, -(1 + 2**-8), -(1 + 3 * 2**-8)], dtype=np.float32
        )
        self.assertEqual(bf16_bits(source).tolist(), [0x3F80, 0x3F82, 0xBF80, 0xBF82])

    def test_scaled_weights_and_preserved_bf16(self):
        with tempfile.TemporaryDirectory() as directory:
            source, destination = Path(directory) / "source", Path(directory) / "output"
            source.mkdir()
            header, payload, expected = {}, bytearray(), {}
            for index in range(20):
                name = f"layer{index}.weight_scale"
                content = struct.pack("<f", 2 ** (index % 5 - 4))
                header[name] = {
                    "dtype": "F32",
                    "shape": [],
                    "data_offsets": [len(payload), len(payload) + len(content)],
                }
                payload.extend(content)
                name = f"layer{index}.weight"
                content = bytes([0, 128, 56, 184, 126, 254])
                header[name] = {
                    "dtype": "F8_E4M3",
                    "shape": [6],
                    "data_offsets": [len(payload), len(payload) + len(content)],
                }
                payload.extend(content)
                values = [0.0, -0.0, 1.0, -1.0, 448.0, -448.0]
                expected[name] = b"".join(
                    struct.pack("<f", value * 2 ** (index % 5 - 4))[2:]
                    for value in values
                )
            for index in range(61):
                name, content = f"kept{index}", struct.pack("<H", 0x3F80 + index)
                header[name] = {
                    "dtype": "BF16",
                    "shape": [1],
                    "data_offsets": [len(payload), len(payload) + 2],
                }
                payload.extend(content)
                expected[name] = content
            encoded = json.dumps(header).encode()
            checkpoint = struct.pack("<Q", len(encoded)) + encoded + payload
            (source / "model.safetensors").write_bytes(checkpoint)
            (source / "manifest.json").write_text(
                json.dumps(
                    {
                        "files": {
                            "model.safetensors": {
                                "sha256": hashlib.sha256(checkpoint).hexdigest()
                            }
                        }
                    }
                )
            )
            (source / "config.json").write_text(
                json.dumps(
                    {
                        "dtype": "bfloat16",
                        "quantization_config": {"quant_method": "compressed-tensors"},
                    }
                )
            )
            with contextlib.redirect_stdout(io.StringIO()):
                convert(source, destination)
            result = (destination / "model.safetensors").read_bytes()
            size = struct.unpack("<Q", result[:8])[0]
            tensors = json.loads(result[8 : 8 + size])
            self.assertEqual(set(tensors), set(expected))
            for name, tensor in tensors.items():
                begin, end = tensor["data_offsets"]
                self.assertEqual(tensor["dtype"], "BF16")
                self.assertEqual(
                    result[8 + size + begin : 8 + size + end], expected[name]
                )
            self.assertNotIn(
                "quantization_config",
                json.loads((destination / "config.json").read_text()),
            )
