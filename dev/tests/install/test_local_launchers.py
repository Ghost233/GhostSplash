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
from unittest import mock

from install import local, paths
from server import server

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = sorted(ROOT.glob("*splash.sh"))
GGUF_SCRIPTS = [script for script in SCRIPTS if "<<'PY'" in script.read_text()]


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
        (self.root / "build").mkdir()
        (self.root / "build/splash").touch()
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
        kv_format="int8",
        startup_options=(),
        experimental_quantized_ffn=None,
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
        converted_draft = script.name in {
            "qwen3827bmlx4bitsplash.sh",
            "swift15flashnextsplash.sh",
        }
        draft_filename = (
            "Qwen3.8-Flash-Next-DFlash2-Q8_0.gguf"
            if script.name == "swift15flashnextsplash.sh"
            else "dflash2-qwen38-27b-Q4_K_M.gguf"
        )
        if converted_draft:
            draft_dir.mkdir(exist_ok=True)
            (draft_dir / "model.safetensors").touch()
            (draft_dir / "config.json").write_text("{}")
            (draft_dir / "conversion.json").write_text(
                json.dumps(
                    {
                        "source_gguf": str(self.target / draft_filename),
                    }
                )
            )
        arguments = [str(self.root), str(self.target)]
        if converted_draft:
            arguments.append(str(draft_dir))
        if converted_draft:
            arguments.append(str(self.target / draft_filename))
        arguments.append(model_id)
        if script.name == "tielminiplus35ba3bsplash.sh":
            arguments.append(str(self.template))
        arguments.extend(["127.0.0.1", "18004", "262144", "40G", "50G", "4", kv_format])
        small_qwen = script.name in {"qwen3.54bsplash.sh", "qwen3.54bdflashsplash.sh"}
        if small_qwen:
            arguments.extend(
                [str(self.root / "build/splash"), experimental_quantized_ffn or "on"]
            )
        elif script.name == "tielminiplus35ba3bsplash.sh":
            arguments.append(experimental_quantized_ffn or "off")
        if script.name in {
            "swift1527biq3ssplash.sh",
            "tielcoder35ba3bq4xlsplash.sh",
            "d3litesplash.sh",
            "d3minisplash.sh",
            "qwen3827bmlx4bitsplash.sh",
            "swift15flashnextsplash.sh",
        }:
            arguments.extend([str(self.root / "build/splash"), *startup_options])
        with (
            mock.patch.object(sys, "argv", [str(script), *arguments]),
            mock.patch.object(paths, "BINARY", paths.BINARY),
            mock.patch("socket.socket"),
            mock.patch.object(os, "chdir"),
            mock.patch.object(
                local, "prepare_local_model", return_value=self.installed
            ) as prepare,
            mock.patch(
                "huggingface_hub.snapshot_download", return_value=str(draft_dir)
            ),
            mock.patch.object(os, "execv", side_effect=ServerStarted) as execute,
            contextlib.redirect_stdout(io.StringIO()),
        ):
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
            self.assertEqual(
                parsed.experimental_quantized_ffn,
                experimental_quantized_ffn or ("on" if small_qwen else "off"),
            )
            self.assertTrue(parsed.no_webui)
            self.assertNotIn("--tokenizer", command)
            prepare.assert_called_once()
            target_names = {
                "qwen3.54bsplash.sh": "Qwen3.5-4B-UD-Q4_K_XL.gguf",
                "qwen3.54bdflashsplash.sh": "Qwen3.5-4B-UD-Q4_K_XL.gguf",
                "swift1527biq3ssplash.sh": "Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf",
                "swift15flashnextsplash.sh": "Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf",
                "tielminiplus35ba3bsplash.sh": "Tiel-Coder-35B-A3B.APEX-I-MiniPlus-V2.1.gguf",
                "tielcoder35ba3bq4xlsplash.sh": "Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf",
            }
            expected_target = (
                self.target / target_names[script.name]
                if script.name in target_names
                else self.target
            )
            self.assertEqual(Path(prepare.call_args.args[0]), expected_target)
            self.assertTrue(prepare.call_args.kwargs["language_only"])
            self.assertEqual(
                prepare.call_args.kwargs["models_dir"],
                self.root / "install/models-local",
            )
            explicit = script.name in {
                "qwen3.54bdflashsplash.sh",
                "qwen3827bmlx4bitsplash.sh",
                "tielminiplus35ba3bsplash.sh",
                "tielcoder35ba3bq4xlsplash.sh",
                "swift15flashnextsplash.sh",
            }
            if explicit:
                self.assertEqual(Path(prepare.call_args.kwargs["draft"]), draft_dir)
            else:
                self.assertNotIn("draft", prepare.call_args.kwargs)
            if script.name == "tielminiplus35ba3bsplash.sh":
                template = parsed.chat_template.read_text()
                self.assertIn("<function=", template)
                self.assertNotIn("System message must be at the beginning.", template)
            else:
                self.assertIsNone(parsed.chat_template)
        return parsed

    def test_swift_forwards_independent_startup_switches_and_paths_with_spaces(self):
        binary = self.root / "alternate binary"
        binary.touch()
        parsed = self.launch(
            ROOT / "swift1527biq3ssplash.sh",
            startup_options=(
                "--binary",
                str(binary),
                "--ane-ffn-share",
                "0.5",
                "--ane-ffn-minimum-rows",
                "512",
                "--ane-power-reference",
                "6",
                "--ane-power-recovery",
                "on",
                "--ane-recovery-max-power-percent",
                "50",
                "--prompt-lookup",
                "off",
                "--compact-draft-head",
                "off",
                "--fast-topk",
                "on",
                "--compact-draft-ranking",
                str(self.root / "rank file"),
            ),
        )
        self.assertEqual(parsed.binary, str(binary))
        self.assertEqual(parsed.ane_ffn_share, 0.5)
        self.assertEqual(parsed.ane_ffn_minimum_rows, 512)
        self.assertEqual(parsed.ane_power_reference, "6.0")
        self.assertEqual(parsed.ane_power_recovery, "on")
        self.assertEqual(parsed.ane_recovery_max_power_percent, 50)
        self.assertEqual(parsed.prompt_lookup, "off")
        self.assertEqual(parsed.compact_draft_head, "off")
        self.assertEqual(parsed.fast_topk, "on")
        self.assertEqual(parsed.compact_draft_ranking, self.root / "rank file")

    def test_saluki_launcher_is_removed(self):
        self.assertFalse((ROOT / "saluki27biq2mixsplash.sh").exists())

    def test_flash_next_uses_ssd_ple_and_resident_experts(self):
        script = ROOT / "swift15flashnextsplash.sh"
        args = self.launch(script)
        self.assertIsNone(args.ple_memory)
        self.assertIsNone(args.expert_memory)
        self.assertEqual(args.expert_ple_memory, 10 * 1024**3)
        self.assertEqual(args.expert_ple_policy, "adaptive")
        self.assertTrue(args.disable_ane)
        self.assertEqual(args.prompt_lookup, "off")
        self.assertEqual(args.fast_topk, "off")
        self.assertEqual(args.compact_draft_head, "off")
        changed = self.launch(
            script,
            startup_options=(
                "--expert-ple-policy",
                "static",
                "--ple-memory",
                "256M",
                "--expert-memory",
                "off",
            ),
        )
        self.assertEqual(changed.ple_memory, 256 * 1024**2)
        self.assertEqual(changed.expert_memory, "off")

    def test_flash_next_rejects_q4_0_sparse_kv(self):
        for kind in ("q4_0", "q4_0_unrotated"):
            with (
                self.subTest(kind=kind),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                self.launch(ROOT / "swift15flashnextsplash.sh", kv_format=kind)
            self.assertEqual(error.exception.code, 2)

    def test_small_qwen_launchers_can_disable_quantized_ffn(self):
        for name in ("qwen3.54bsplash.sh", "qwen3.54bdflashsplash.sh"):
            with self.subTest(launcher=name):
                self.launch(ROOT / name, experimental_quantized_ffn="off")

    def test_shell_quantized_ffn_arguments_ignore_environment_and_validate_values(self):
        command = [
            "sh",
            "-c",
            '. "$1"; shift; launcher_allow_baseline=1; '
            'splash_parse_launcher_options "$@"; echo "$experimental_quantized_ffn"',
            "launcher-test",
            str(ROOT / "dev/tools/launcher_kv.sh"),
        ]
        environment = dict(os.environ, experimental_quantized_ffn="on")
        for args, code, expected in (
            ([], 0, "off"),
            (["--experimental-quantized-ffn", "on"], 0, "on"),
            (["--experimental-quantized-ffn=off"], 0, "off"),
            (["--experimental-quantized-ffn"], 2, "requires on or off"),
            (["--experimental-quantized-ffn", "invalid"], 2, "requires on or off"),
            (
                ["--baseline", "--experimental-quantized-ffn", "on"],
                2,
                "--baseline requires --experimental-quantized-ffn off",
            ),
        ):
            with self.subTest(args=args):
                result = subprocess.run(
                    [*command, *args], capture_output=True, text=True, env=environment
                )
                self.assertEqual(result.returncode, code, result.stderr)
                self.assertIn(expected, result.stdout + result.stderr)

    def test_mlx_27b_uses_local_draft_and_forwards_startup_options(self):
        script = ROOT / "qwen3827bmlx4bitsplash.sh"
        parsed = self.launch(script)
        self.assertEqual(parsed.prompt_lookup, "on")
        self.assertEqual(parsed.compact_draft_head, "off")
        self.assertIsNone(parsed.compact_draft_ranking)
        self.assertEqual(parsed.ane_ffn_share, 0.4182)
        self.assertEqual(parsed.ane_power_reference, "auto")
        self.assertEqual(parsed.ane_power_recovery, "on")
        self.assertEqual(parsed.ane_recovery_max_power_percent, 50)
        self.assertEqual(parsed.fast_topk, "on")
        parsed = self.launch(
            script,
            startup_options=(
                "--disable-ane",
                "--prompt-lookup",
                "off",
                "--compact-draft-head",
                "off",
                "--fast-topk",
                "off",
                "--ane-ffn-share",
                "auto",
                "--ane-power-recovery",
                "off",
                "--ane-power-reference",
                "off",
            ),
        )
        self.assertTrue(parsed.disable_ane)
        self.assertEqual(parsed.prompt_lookup, "off")
        self.assertEqual(parsed.compact_draft_head, "off")
        self.assertEqual(parsed.fast_topk, "off")
        self.assertIsNone(parsed.ane_ffn_share)
        self.assertEqual(parsed.ane_power_reference, "off")
        self.assertEqual(parsed.ane_power_recovery, "off")

    def test_modern_launchers_default_compact_off_and_require_explicit_public_ranking(
        self,
    ):
        for name in (
            "swift1527biq3ssplash.sh",
            "d3litesplash.sh",
            "d3minisplash.sh",
        ):
            with self.subTest(launcher=name):
                parsed = self.launch(ROOT / name)
                self.assertEqual(parsed.compact_draft_head, "off")
                self.assertIsNone(parsed.compact_draft_ranking)
                parsed = self.launch(
                    ROOT / name,
                    startup_options=(
                        "--compact-draft-ranking",
                        str(self.root / "public ranking.splash"),
                    ),
                )
                self.assertEqual(parsed.compact_draft_head, "off")

    def test_every_gguf_launcher_first_install_and_server_arguments(self):
        for script in GGUF_SCRIPTS:
            with self.subTest(script=script.name):
                self.launch(script)

    def test_every_gguf_launcher_forwards_each_kv_format(self):
        for script in GGUF_SCRIPTS:
            formats = (
                ("int8", "bf16", "q4h")
                if script.name == "swift15flashnextsplash.sh"
                else ("int8", "bf16", "q4_0", "q4_0_unrotated", "q4h")
            )
            for kv_format in formats:
                with self.subTest(script=script.name, kv_format=kv_format):
                    self.launch(script, kv_format=kv_format)

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
                    if script.name
                    in {
                        "qwen3.54bsplash.sh",
                        "qwen3.54bdflashsplash.sh",
                        "d3litesplash.sh",
                        "d3minisplash.sh",
                    }
                    else "0.0.0.0",
                )
                with socket.socket() as listener:
                    listener.bind((host, 0))
