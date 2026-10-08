"""Flash Next source configuration and split selection, without payload reads."""

import unittest
from types import SimpleNamespace

from install import assembly, families, gguf, models, upstream


class FlashNextSourceTests(unittest.TestCase):
    def test_selects_first_of_complete_iq3s_pair(self):
        names = {
            "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
            "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf",
            "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf",
            "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf",
        }
        first, _ = upstream.select_gguf(names, "IQ3_S")
        self.assertEqual(
            first, "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"
        )
        self.assertEqual(len(upstream.gguf_shards(names, first)), 2)

    def test_missing_second_shard_is_refused(self):
        first = "Swift-Flash-Next-IQ3_S-00001-of-00002.gguf"
        with self.assertRaisesRegex(models.ModelError, "missing.*shard"):
            upstream.gguf_shards({first}, first)

    def test_unsplit_selection_preserved(self):
        names = {"Qwen-Q4_K_M.gguf", "Qwen-IQ3_S.gguf"}
        self.assertEqual(upstream.select_gguf(names, "IQ3_S")[0], "Qwen-IQ3_S.gguf")
        self.assertEqual(
            upstream.gguf_shards(names, "Qwen-IQ3_S.gguf"), ["Qwen-IQ3_S.gguf"]
        )

    def test_qwen4exp_declares_distinct_architecture(self):
        self.assertEqual(gguf.TEXT_MODEL_TYPES["qwen4exp"], "qwen4_exp_text")

    def test_draftless_family_never_resolves_another_models_draft(self):
        family = families.named("Qwen3.8-Flash-Next")
        self.assertIsNone(family.draft_repo)
        selection = SimpleNamespace(draft_model=None)
        self.assertIsNone(upstream._resolve_draft(family, selection, None, None))
        self.assertEqual(
            upstream._draft(SimpleNamespace(family=family), None, None), (None, {})
        )
        selection.draft_model = "incoai/Qwen3.8-27B-DFlash2"
        with self.assertRaisesRegex(models.ModelError, "external draft"):
            upstream._resolve_draft(family, selection, None, None)

    def test_only_flash_text_gguf_can_record_no_draft(self):
        record = {
            "version": 1,
            "model": "owner/model",
            "family": "Qwen3.8-Flash-Next",
            "target_format": "gguf",
            "vision_format": "none",
            "metadata": "a" * 64,
            "sources": {
                "target": {"repo": "owner/model", "revision": "a" * 40},
                "draft": None,
            },
            "files": {},
        }
        self.assertTrue(assembly._well_formed(record))
        self.assertEqual(assembly.pins(record), [])
        record["family"] = "Qwen3.8-27B"
        self.assertFalse(assembly._well_formed(record))
        record["family"] = "Qwen3.8-Flash-Next"
        record["vision_format"] = "gguf"
        self.assertFalse(assembly._well_formed(record))


if __name__ == "__main__":
    unittest.main()
