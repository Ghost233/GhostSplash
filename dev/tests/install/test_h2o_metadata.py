import unittest

from dev.tests.install import test_kev_metadata as fixture
from dev.tests.install.test_gguf_metadata import fixture as tokenizer_fixture
from install import gguf


class H2OMetadataTests(unittest.TestCase):
    metadata = fixture.KevMetadataTests.metadata

    def values(self):
        return fixture.kev_metadata() | {
            "qwen35.decision.type": "h2o-lightning",
            "qwen35.decision.labels": [" A", " α"],
            "tokenizer.ggml.tokens": ["ĠA", "ĠÎ±"],
        }

    def test_h2o_preserves_labels_temperatures_and_tied_readout(self):
        header = self.metadata(self.values())
        config = gguf.model_config(header)["decision_config"]
        self.assertEqual(config["type"], "h2o-lightning")
        self.assertEqual(config["codes"], [" A", " α"])
        self.assertEqual(config["token_ids"], [0, 1])
        self.assertEqual(config["temperatures"]["choice"], 2.406049966812134)
        required = gguf.loaded_tensors(header)
        self.assertNotIn("output.weight", required)
        self.assertNotIn("cls.output.weight", required)
        header.tensors = {
            name: 0 if "F32" in types else 8 for name, types in required.items()
        }
        header.extents = {"token_embd.weight": (2560, 2)}
        gguf.require_loadable(header)

    def test_invalid_h2o_labels_and_temperature_are_rejected(self):
        for key, value in (
            ("qwen35.decision.labels", []),
            ("qwen35.decision.labels", [" A", " A"]),
            ("qwen35.decision.labels", [" A", " absent"]),
            ("qwen35.decision.temperature.choice", 0.0),
        ):
            with self.subTest(key=key, value=value):
                with self.assertRaises(gguf.ModelError):
                    gguf.model_config(self.metadata(self.values() | {key: value}))

    def test_qwen2_tokenizer_honors_the_gguf_normalization_flag(self):
        from tokenizers import Tokenizer

        for nfc in (False, True):
            header = self.metadata(
                tokenizer_fixture()
                | {
                    "tokenizer.ggml.pre": "qwen2",
                    "tokenizer.ggml.normalizer.nfc": nfc,
                }
            )
            backend = Tokenizer.from_str(
                gguf.tokenizer_files(header)["tokenizer/tokenizer.json"].decode()
            )
            self.assertEqual(
                backend.encode("é").ids == backend.encode("e\u0301").ids, nfc
            )

    def test_qwen2_groups_combining_marks_with_punctuation(self):
        from tokenizers import Tokenizer, decoders

        header = self.metadata(tokenizer_fixture() | {"tokenizer.ggml.pre": "qwen2"})
        backend = Tokenizer.from_str(
            gguf.tokenizer_files(header)["tokenizer/tokenizer.json"].decode()
        )
        # H2O's official unicode-06 probe keeps these combining marks in
        # punctuation runs, rather than treating them as letters.
        pieces = [
            decoders.ByteLevel().decode([piece])
            for piece, _ in backend.pre_tokenizer.pre_tokenize_str("है? दिनांक।")
        ]
        self.assertEqual(pieces, ["ह", "ै?", " द", "िन", "ां", "क", "।"])
