"""Standard 4B draft selection stays explicit and pins the audited release.

Configuration semantics are checked by the native model-check tests; these
cases exercise installer resolution, downloaded files and assembly identity.
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from dev.tests.installer_fixtures import FakeHub, draft_dir, selection
from install import assembly, families, hub, models, upstream

FAMILY = families.named("Qwen3.5-4B")
REPO = families.QWEN35_4B_DFLASH_REPO
REVISION = families.QWEN35_4B_DFLASH_REVISION


class StandardDFlashTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.target_repo = hub.Repository(
            "someone/4b-target", "a" * 40, frozenset({"target.gguf"})
        )
        self.target = SimpleNamespace(
            family=FAMILY,
            format="gguf",
            vision_format="none",
            config=self.root / "target-config.json",
            gguf_metadata=self.root / "gguf-metadata.json",
        )

    def test_default_4b_has_no_draft_or_draft_hub_request(self):
        self.assertIsNone(FAMILY.draft_repo)
        with mock.patch.object(hub.Repository, "resolve") as resolve:
            draft = upstream._resolve_draft(
                FAMILY, selection(self.root), None, self.target_repo
            )
        resolve.assert_not_called()
        self.assertIsNone(draft)
        self.assertEqual(upstream._draft(self.target, None, draft), (None, {}))

    def test_explicit_local_standard_checkpoint_is_downloaded_and_checked(self):
        source = draft_dir(self.root / "draft", FAMILY)
        chosen = selection(self.root, draft_model=str(source))
        draft = upstream._resolve_draft(FAMILY, chosen, None, self.target_repo)
        with mock.patch.object(upstream, "check_model", return_value=FAMILY) as check:
            selected, files = upstream._draft(self.target, None, draft)
        self.assertEqual(selected.identity(), {"repo": str(source), "revision": None})
        self.assertEqual(set(files), {"draft/config.json", "draft/model.safetensors"})
        check.assert_called_once_with(
            "gguf",
            "none",
            self.target.config,
            gguf_metadata=self.target.gguf_metadata,
            draft=source / "config.json",
        )

    def test_explicit_official_standard_release_uses_fixed_revision(self):
        fake = FakeHub(self, self.root / "hub")
        fake.publish(REPO, REVISION, lambda path: draft_dir(path, FAMILY))
        # A different main revision must not replace the requested checkpoint.
        fake.publish(REPO, "e" * 40, lambda path: draft_dir(path, FAMILY))
        chosen = selection(self.root, draft_model=REPO)
        draft = upstream._resolve_draft(FAMILY, chosen, None, self.target_repo)
        self.assertEqual(fake.requests, [(REPO, REVISION)])
        self.assertEqual(draft.revision, REVISION)
        with mock.patch.object(upstream, "check_model", return_value=FAMILY):
            selected, files = upstream._draft(self.target, None, draft)
        self.assertEqual(selected.identity(), {"repo": REPO, "revision": REVISION})
        self.assertEqual(set(files), {"draft/config.json", "draft/model.safetensors"})
        installed = {"sources": {"draft": None}, "target_format": "mlx-affine"}
        self.assertEqual(
            upstream._changes(installed, FAMILY, draft),
            [f"its draft is now {REPO}@{REVISION[:12]}"],
        )

    def test_prior_main_commit_cannot_substitute_for_pinned_standard_release(self):
        fake = FakeHub(self, self.root / "hub")
        fake.publish(REPO, REVISION, lambda path: draft_dir(path, FAMILY))
        installed = {"sources": {"draft": {"repo": REPO, "revision": "e" * 40}}}
        chosen = selection(self.root, draft_model=REPO)
        draft = upstream._resolve_draft(FAMILY, chosen, installed, self.target_repo)
        self.assertEqual(fake.requests, [(REPO, REVISION)])
        self.assertEqual(draft.revision, REVISION)

    def test_incompatible_explicit_draft_is_refused_before_weight_download(self):
        fake = FakeHub(self, self.root / "hub")
        fake.publish(REPO, REVISION, lambda path: draft_dir(path, FAMILY))
        draft = upstream._resolve_draft(
            FAMILY, selection(self.root, draft_model=REPO), None, self.target_repo
        )
        with (
            mock.patch.object(
                upstream,
                "check_model",
                side_effect=models.ModelError("draft shape mismatch"),
            ),
            self.assertRaisesRegex(
                models.ModelError, "draft configuration is incompatible with Qwen3.5-4B"
            ),
        ):
            upstream._draft(self.target, None, draft)
        self.assertEqual(fake.downloads, [f"{REPO}/config.json"])

    def test_assembly_round_trips_no_draft_and_explicit_local_draft(self):
        source = draft_dir(self.root / "draft", FAMILY)
        config = source / "config.json"
        for draft in (None, {"repo": str(source), "revision": None}):
            files = {"config.json": config}
            if draft:
                files |= {
                    "draft/config.json": config,
                    "draft/model.safetensors": source / "model.safetensors",
                }
            record = {
                "version": 1,
                "model": "someone/4b-target",
                "family": FAMILY.name,
                "target_format": "gguf",
                "vision_format": "none",
                "metadata": "0" * 64,
                "sources": {
                    "target": {"repo": "someone/4b-target", "revision": "a" * 40},
                    "draft": draft,
                },
                "files": {
                    name: assembly.file_record(path) for name, path in files.items()
                },
            }
            built = assembly.build(self.root / "models", record, files)
            self.assertEqual(
                assembly.verify(built, full=True)["sources"]["draft"], draft
            )


if __name__ == "__main__":
    unittest.main()
