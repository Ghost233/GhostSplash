import ast
import contextlib
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from install import assembly, hub, models, upstream
from server import server

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = sorted(ROOT.glob("*splash.sh"))
GGUF_SCRIPTS = [script for script in SCRIPTS if "<<'PY'" in script.read_text()]
EXPLICIT_DRAFTS = {
    "ornith1535ba3bsplash.sh",
    "tiel35ba3bsplash.sh",
    "tielminiplus35ba3bsplash.sh",
    "qwen38distill35ba3bsplash.sh",
}


class ServerStarted(Exception):
    pass


class LocalLauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="splash launcher ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.target = self.root / "target"
        self.target.mkdir()
        (self.target / "model.gguf").touch()
        self.installed = self.root / "installed"
        (self.installed / "tokenizer").mkdir(parents=True)
        self.template = self.target / "chat_template.jinja"
        self.template.write_text(
            (ROOT / "docs/templates/tiel-miniplus-v2.1.original.jinja").read_text()
        )

    def launch(
        self,
        script,
        *,
        installed=False,
        invalid=False,
        wrong_target=False,
        kv_format="int8",
    ):
        text = script.read_text()
        source = text.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
        source_line = text.split("<<'PY'\n", 1)[0].count("\n") + 1
        tree = ast.increment_lineno(ast.parse(source), source_line)
        model_id = "community/local-model"
        draft_dir = self.root / "draft"
        if script.name == "qwen3.54bdflashsplash.sh":
            draft_dir = (
                self.root / "install/models-local/drafts/qwen35-4b-dflash/source"
            )
            draft_dir.mkdir(parents=True, exist_ok=True)
            (draft_dir / "model.safetensors").touch()
        arguments = [str(self.root), str(self.target)]
        if script.name == "coder39027bsplash.sh":
            arguments.append(str(draft_dir))
        arguments.append(model_id)
        if script.name == "tielminiplus35ba3bsplash.sh":
            arguments.append(str(self.template))
        arguments.extend(["127.0.0.1", "18004", "262144", "40G", "50G", "4", kv_format])
        record = {
            "sources": {
                "target": {
                    "repo": "wrong" if wrong_target else str(self.target),
                    "revision": None,
                },
                "draft": None
                if script.name == "qwen3.54bsplash.sh"
                else {"repo": str(draft_dir), "revision": None},
            }
        }
        scratches = []

        def inspect(repo, variant, language_only, scratch):
            self.assertEqual(repo.name, str(self.target))
            self.assertIsInstance(scratch, Path)
            self.assertTrue(scratch.is_dir())
            (scratch / "config.json").write_text("{}")
            scratches.append(scratch)
            if invalid:
                raise models.ModelError("invalid target")

        with (
            mock.patch.object(sys, "argv", [str(script), *arguments]),
            mock.patch("socket.socket"),
            mock.patch.object(os, "chdir"),
            mock.patch.object(os, "set_inheritable") as inherit,
            mock.patch.object(
                models,
                "installation_kind",
                return_value=models.ASSEMBLY if installed else None,
            ),
            mock.patch.object(upstream, "_gguf_target", side_effect=inspect),
            mock.patch.object(hub.Repository, "resolve", autospec=True) as draft,
            mock.patch.object(
                upstream,
                "_install",
                autospec=True,
                side_effect=models.ModelError("invalid target") if invalid else None,
            ) as install,
            mock.patch.object(assembly, "verify", return_value=record) as verify,
            mock.patch.object(
                assembly,
                "hold",
                return_value=(self.installed, SimpleNamespace(fileno=lambda: 99)),
            ) as hold,
            mock.patch.object(os, "execv", side_effect=ServerStarted) as execute,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            if invalid or wrong_target:
                with self.assertRaises(models.ModelError):
                    exec(compile(tree, str(script), "exec"), {})
                execute.assert_not_called()
                hold.assert_not_called()
                draft.assert_not_called()
            else:
                with self.assertRaises(ServerStarted):
                    exec(compile(tree, str(script), "exec"), {})
                command = execute.call_args.args[1]
                self.assertEqual(
                    command[:4],
                    [str(self.root / ".venv/bin/python"), "-u", "-m", "server.server"],
                )
                parsed = server.parse_args(command[4:])
                self.assertEqual(parsed.model_root, str(self.installed))
                self.assertEqual(parsed.model, model_id)
                self.assertEqual(parsed.host, "127.0.0.1")
                self.assertEqual(parsed.port, 18004)
                self.assertEqual(parsed.max_context, 262144)
                self.assertEqual(parsed.queue_size, 4)
                self.assertEqual(parsed.kv_format, kv_format)
                self.assertTrue(parsed.no_webui)
                inherit.assert_called_once_with(99, True)
                if script.name == "tielminiplus35ba3bsplash.sh":
                    template = (
                        Path(parsed.tokenizer) / "chat_template.jinja"
                    ).read_text()
                    self.assertIn("<function=", template)
                    self.assertNotIn(
                        "System message must be at the beginning.", template
                    )
            if installed:
                verify.assert_called_once()
                install.assert_not_called()
                draft.assert_not_called()
            elif not invalid:
                install.assert_called_once()
                if script.name == "qwen3.54bdflashsplash.sh":
                    chosen = install.call_args.args[0]
                    self.assertEqual(chosen.draft_model, str(draft_dir))
                    self.assertEqual(
                        chosen.models_root,
                        (self.root / "install/models-local").resolve(),
                    )
                    installed_draft = install.call_args.kwargs["draft"]
                    self.assertEqual(installed_draft.directory, draft_dir)
                    self.assertEqual(
                        installed_draft.identity(),
                        {"repo": str(draft_dir), "revision": None},
                    )
                    self.assertIn("model.safetensors", installed_draft.files)
                    draft.assert_not_called()
            if script.name in EXPLICIT_DRAFTS and not installed:
                self.assertEqual(len(scratches), 1)
            self.assertTrue(all(not scratch.exists() for scratch in scratches))

    def test_every_gguf_launcher_first_install_and_server_arguments(self):
        for script in GGUF_SCRIPTS:
            with self.subTest(script=script.name):
                self.launch(script)

    def test_every_gguf_launcher_forwards_each_kv_format(self):
        for script in GGUF_SCRIPTS:
            for kv_format in ("int8", "bf16", "q4_0", "q4_0_unrotated", "q4h"):
                with self.subTest(script=script.name, kv_format=kv_format):
                    self.launch(script, installed=True, kv_format=kv_format)

    def test_every_launcher_listens_without_a_specific_virtual_bridge_address(self):
        for script in SCRIPTS:
            with self.subTest(script=script.name):
                host = next(
                    line.split("=", 1)[1]
                    for line in script.read_text().splitlines()
                    if line.startswith("host=")
                )
                self.assertEqual(
                    host,
                    "127.0.0.1"
                    if script.name in {"qwen3.54bsplash.sh", "qwen3.54bdflashsplash.sh"}
                    else "0.0.0.0",
                )
                with socket.socket() as listener:
                    listener.bind((host, 0))

    def test_every_gguf_launcher_reuses_an_installation_without_downloads(self):
        for script in GGUF_SCRIPTS:
            with self.subTest(script=script.name):
                self.launch(script, installed=True)

    def test_every_gguf_launcher_refuses_another_targets_installation(self):
        for script in GGUF_SCRIPTS:
            with self.subTest(script=script.name):
                self.launch(script, installed=True, wrong_target=True)

    def test_explicit_draft_launchers_refuse_target_before_resolving_draft(self):
        for script in GGUF_SCRIPTS:
            if script.name in EXPLICIT_DRAFTS:
                with self.subTest(script=script.name):
                    self.launch(script, invalid=True)

    def test_legacy_launchers_accept_packages_without_layout_json(self):
        for script in SCRIPTS:
            if script in GGUF_SCRIPTS:
                continue
            with self.subTest(script=script.name):
                repo = self.root / script.stem
                repo.mkdir()
                target = repo / "model"
                target.mkdir()
                (target / "manifest.json").write_text("{}")
                for name in ("target", "draft", "tokenizer", "vision"):
                    (target / name).mkdir()
                (repo / "dev/tools").mkdir(parents=True)
                (repo / "dev/tools/launcher_kv.sh").write_text(
                    (ROOT / "dev/tools/launcher_kv.sh").read_text()
                )
                (repo / "server").mkdir()
                (repo / "server/server.py").touch()
                (repo / "build").mkdir()
                (repo / "build/splash").touch()
                (repo / ".venv/bin").mkdir(parents=True)
                output = repo / "arguments.json"
                capture = repo / "capture.py"
                capture.write_text(
                    "import json, pathlib, sys\n"
                    f"pathlib.Path({str(output)!r}).write_text(json.dumps(sys.argv[1:]))\n"
                )
                python = repo / ".venv/bin/python"
                python.write_text(
                    f'#!/bin/sh\nexec "{sys.executable}" "{capture}" "$@"\n'
                )
                python.chmod(0o755)
                tools = repo / "tools"
                tools.mkdir()
                for name, result in (("lsof", 1), ("curl", 0), ("open", 0)):
                    tool = tools / name
                    tool.write_text(f"#!/bin/sh\nexit {result}\n")
                    tool.chmod(0o755)
                launcher = repo / script.name
                lines = script.read_text().splitlines()
                launcher.write_text(
                    "\n".join(
                        f'model_dir="{target}"'
                        if line.startswith("model_dir=")
                        else line
                        for line in lines
                    )
                    + "\n"
                )
                completed = subprocess.run(
                    ["sh", str(launcher), "--kv-format", "q4_0"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    env={
                        **os.environ,
                        "PATH": str(tools) + os.pathsep + os.environ["PATH"],
                    },
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                command = json.loads(output.read_text())
                self.assertEqual(command[:3], ["-u", "-m", "server.server"])
                parsed = server.parse_args(command[3:])
                self.assertEqual(parsed.model_root, str(target))
                self.assertEqual(parsed.tokenizer, str(target / "tokenizer"))
                self.assertEqual(parsed.kv_format, "q4_0")
                self.assertTrue(parsed.no_webui)
