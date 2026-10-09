"""Flash Next source configuration and split selection, without payload reads."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from dev.tests.install.test_gguf_metadata import fixture, loadable_tensors, write_gguf
from install import assembly, families, gguf, hub, models, upstream


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

    def test_split_architecture_may_be_declared_only_in_first_shard(self):
        for repeated_architecture in (None, "qwen35moe"):
            with self.subTest(architecture=repeated_architecture):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    values = fixture()
                    tensors = list(loadable_tensors(values, root).items())
                    first = root / "Model-IQ3_S-00001-of-00002.gguf"
                    second = root / "Model-IQ3_S-00002-of-00002.gguf"
                    split = {"split.count": 2, "split.tensors.count": len(tensors)}
                    write_gguf(first, values | split | {"split.no": 0}, tensors[:1])
                    later = split | {"split.no": 1}
                    if repeated_architecture is not None:
                        later["general.architecture"] = repeated_architecture
                    write_gguf(second, later, tensors[1:])
                    repo = hub.Repository.local_directory(root)
                    with mock.patch.object(
                        upstream,
                        "check_model",
                        return_value=families.named("Qwen3.6-35B-A3B"),
                    ):
                        target = upstream.inspect_target(repo, "IQ3_S", True, root)
                    self.assertEqual(len(target.files), 2)
                    self.assertEqual(
                        models.read_json(target.config)["text_config"]["model_type"],
                        "qwen3_5_moe_text",
                    )
                    self.assertIn(
                        "tokenizer/tokenizer.json",
                        gguf.derived_files(first, shards=[second]),
                    )

    def test_split_architecture_conflict_is_still_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "Model-IQ3_S-00001-of-00002.gguf"
            second = root / "Model-IQ3_S-00002-of-00002.gguf"
            split = {"split.count": 2, "split.tensors.count": 0}
            write_gguf(first, fixture() | split | {"split.no": 0})
            write_gguf(
                second,
                split | {"split.no": 1, "general.architecture": "qwen35"},
            )
            repo = hub.Repository.local_directory(root)
            with self.assertRaisesRegex(models.ModelError, "different architectures"):
                upstream.inspect_target(repo, "IQ3_S", True, root)
            with self.assertRaisesRegex(
                models.ModelError, "invalid GGUF metadata shard"
            ):
                gguf.derived_files(first, shards=[second])

    def test_draftless_family_never_resolves_another_models_draft(self):
        family = families.named("Qwen3.8-Flash-Next")
        self.assertIsNone(family.draft_repo)
        selection = SimpleNamespace(draft_model=None)
        self.assertIsNone(upstream._resolve_draft(family, selection, None, None))
        self.assertEqual(
            upstream._draft(SimpleNamespace(family=family), None, None), (None, {})
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection.draft_model = str(root)
            (root / "config.json").write_bytes(
                models.json_bytes({"model_type": "DFlash2DraftModel"})
            )
            with self.assertRaisesRegex(
                models.ModelError, "external draft.*qwen4_exp_mtp"
            ):
                upstream._resolve_draft(family, selection, None, None)
            (root / "config.json").write_bytes(
                models.json_bytes({"model_type": "qwen4_exp_mtp"})
            )
            draft = upstream._resolve_draft(family, selection, None, None)
            self.assertEqual(draft.identity(), {"repo": str(root), "revision": None})

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
