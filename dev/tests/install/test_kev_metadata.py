import copy
import tempfile
import unittest
from pathlib import Path

from dev.tests import fixture_files
from dev.tests.install.test_gguf_metadata import fixture
from install import assembly, gguf


def kev_metadata():
    values = fixture() | {
        "general.architecture": "qwen35",
        "qwen35.block_count": 32,
        "qwen35.embedding_length": 2560,
        "qwen35.attention.head_count": 16,
        "qwen35.attention.head_count_kv": 4,
        "qwen35.attention.key_length": 256,
        "qwen35.context_length": 262144,
        "qwen35.full_attention_interval": 4,
        "qwen35.decision.type": "kev",
        "qwen35.embedding_length_out": 512,
        "tokenizer.chat_template.systemone": "<|fim_prefix|>{{ state }}<|fim_middle|>{{ instructions }}{% for o in options %}<|box_start|>{% if type == 'score' %}{% if o.description %}{{ o.description }}{% endif %}{% else %}{% if type != 'noul' %}{{ o.key }}{% elif o.key == 'true' %}yes{% else %}no{% endif %}{% if o.description %}: {{ o.description }}{% endif %}{% endif %}<|box_end|>{% endfor %}<|fim_suffix|>",
    }
    for kind in ("choice", "score", "noul"):
        values["qwen35.decision.temperature." + kind] = 2.40605
    return values


class KevMetadataTests(unittest.TestCase):
    def metadata(self, values, tensors=()):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return gguf.Metadata(
            fixture_files.write_gguf(
                Path(directory.name) / "model.gguf", values, tensors
            ),
            tensors=True,
        )

    def test_config_preserves_decision_semantics(self):
        header = self.metadata(kev_metadata())
        config = gguf.model_config(header)
        self.assertEqual(config["decision_config"]["type"], "kev")
        self.assertEqual(config["decision_config"]["head_size"], 512)
        self.assertEqual(
            config["decision_config"]["temperatures"]["choice"],
            header.values["qwen35.decision.temperature.choice"],
        )
        self.assertEqual(
            config["decision_config"]["chat_template"],
            header.values["tokenizer.chat_template.systemone"],
        )

    def test_screen_requires_classifier_and_omits_lm_head(self):
        header = self.metadata(kev_metadata())
        required = gguf.loaded_tensors(header)
        self.assertTrue(
            "cls.output.weight" in required, "decision weight was not screened"
        )
        self.assertIn("cls.output.bias", required)
        self.assertNotIn("output.weight", required)
        header.tensors = {
            name: 0 if "F32" in types else 12 for name, types in required.items()
        }
        header.extents = {"cls.output.weight": (2560, 512), "cls.output.bias": (512,)}
        gguf.require_loadable(header)
        for name in ("cls.output.weight", "cls.output.bias"):
            bad = copy.copy(header)
            bad.tensors = dict(header.tensors)
            del bad.tensors[name]
            with self.assertRaisesRegex(gguf.ModelError, "cls.output"):
                gguf.require_loadable(bad)
        for shape in ((512, 2560), (2560, 256)):
            bad = copy.copy(header)
            bad.extents = dict(header.extents, **{"cls.output.weight": shape})
            with self.assertRaisesRegex(gguf.ModelError, "shape"):
                gguf.require_loadable(bad)

    def test_kev_08_accepts_its_actual_pointer_head_width(self):
        values = kev_metadata() | {
            "qwen35.embedding_length": 1024,
            "qwen35.block_count": 24,
            "qwen35.attention.head_count": 8,
            "qwen35.attention.head_count_kv": 2,
        }
        header = self.metadata(values)
        required = gguf.loaded_tensors(header)
        header.tensors = {
            name: 0 if "F32" in types else 8 for name, types in required.items()
        }
        header.extents = {"cls.output.weight": (1024, 512), "cls.output.bias": (512,)}
        gguf.require_loadable(header)
        header.extents["cls.output.weight"] = (2560, 512)
        with self.assertRaisesRegex(gguf.ModelError, "shape"):
            gguf.require_loadable(header)

    def test_kev_assembly_accepts_no_draft_without_relaxing_other_families(self):
        record = {
            "version": 1,
            "model": "local/kev",
            "family": "Kev-4B",
            "target_format": "gguf",
            "vision_format": "none",
            "metadata": "a" * 64,
            "sources": {
                "target": {"repo": "/local/kev", "revision": None},
                "draft": None,
            },
            "files": {},
        }
        self.assertTrue(assembly._well_formed(record))
        for family in ("Qwen3.8-27B", "Qwen3.6-35B-A3B"):
            self.assertFalse(assembly._well_formed(record | {"family": family}))

    def test_invalid_decision_metadata_is_rejected(self):
        for key, value in (
            ("qwen35.decision.type", "unknown"),
            ("qwen35.embedding_length_out", 256),
            ("qwen35.decision.temperature.choice", 0.0),
            ("qwen35.decision.temperature.noul", float("nan")),
            ("tokenizer.chat_template.systemone", ""),
        ):
            with self.subTest(key=key, value=value):
                with self.assertRaises(gguf.ModelError):
                    gguf.model_config(self.metadata(kev_metadata() | {key: value}))
