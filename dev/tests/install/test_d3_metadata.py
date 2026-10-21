import copy
import unittest

from dev.tests.install import test_kev_metadata as kev_fixture
from install import gguf


class D3MetadataTests(unittest.TestCase):
    metadata = kev_fixture.KevMetadataTests.metadata

    def test_d3_geometries_keep_noncausal_and_code_head(self):
        for hidden, layers, heads, kv in ((1024, 24, 8, 2), (2560, 32, 16, 4)):
            values = kev_fixture.kev_metadata() | {
                "qwen35.decision.type": "vllm-d3",
                "qwen35.attention.causal": False,
                "qwen35.embedding_length": hidden,
                "qwen35.block_count": layers,
                "qwen35.attention.head_count": heads,
                "qwen35.attention.head_count_kv": kv,
            }
            # Small fixtures need only two valid answer-code tokens.
            values["tokenizer.ggml.tokens"] = ["A", "B"]
            header = self.metadata(values)
            config = gguf.model_config(header)["decision_config"]
            self.assertEqual(config["type"], "vllm-d3")
            self.assertEqual(config["token_ids"], [0, 1])
            self.assertEqual(config["codes"], ["A", "B"])
            required = gguf.loaded_tensors(header)
            self.assertIn("output.weight", required)
            self.assertNotIn("cls.output.weight", required)
            header.tensors = {
                name: 0 if types == {"F32"} else 30 for name, types in required.items()
            }
            header.extents = {"output.weight": (hidden, 2)}
            gguf.require_loadable(header)
            bad = copy.copy(header)
            bad.values = dict(header.values, **{"qwen35.attention.causal": True})
            with self.assertRaisesRegex(gguf.ModelError, "noncausal"):
                gguf.model_config(bad)


if __name__ == "__main__":
    unittest.main()
