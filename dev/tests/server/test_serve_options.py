import io
import math
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from install import launcher
from server import images, serve_options
from server import server as api
from server.origins import ANY_ORIGIN

ROOT = Path(__file__).resolve().parents[3]
# Each parser, with the arguments it requires.
SERVER_ARGS = ["model", "--tokenizer", "tokenizer", "--model", "owner/repo"]
LAUNCHER_ARGS = ["serve", "--model", "owner/repo"]
PARSERS = ((api.parse_args, SERVER_ARGS), (launcher.parse_args, LAUNCHER_ARGS))


def values(flag, accepted, refused, repeatable=False):
    """Arguments that give `flag` each accepted value, with the value both
    parsers parse it to, and arguments that give it each refused value."""
    return (
        [
            ([f"{flag}={text}"], [parsed] if repeatable else parsed)
            for text, parsed in accepted.items()
        ],
        [[f"{flag}={text}"] for text in refused],
    )


# Every shared option: accepted arguments with the value they parse to, and
# refused arguments.
OPTIONS = {
    "--experimental-quantized-ffn": values(
        "--experimental-quantized-ffn",
        {"off": "off", "on": "on"},
        ("auto", "quantized", "1", ""),
    ),
    "--encoder-option-queries": values(
        "--encoder-option-queries", {"on": "on", "off": "off"}, ("auto", "")
    ),
    "--encoder-submit-batch-size": values(
        "--encoder-submit-batch-size", {"4": 4, "2": 2, "1": 1}, ("0", "3", "5", "auto")
    ),
    "--host": values(
        "--host", {"0.0.0.0": "0.0.0.0", "mymac.local": "mymac.local"}, ()
    ),
    "--served-model-name": values(
        "--served-model-name",
        {"local": "local", "-local": "-local", "org/model": "org/model"},
        ("", "has space", "a?b", "a#b", "a%b", "a\\b", "/a", "a//b", "a/../b"),
        repeatable=True,
    ),
    "--announce-served-name": (
        [(["--announce-served-name", "--served-model-name=local"], True)],
        [["--announce-served-name"]],
    ),
    "--default-reasoning-effort": values(
        "--default-reasoning-effort",
        {effort: effort for effort in serve_options.REASONING_EFFORTS},
        ("", "turbo", "XHIGH"),
    ),
    "--kv-format": values(
        "--kv-format",
        {
            "int8": "int8",
            "bf16": "bf16",
            "q4h": "q4h",
            "q4_0": "q4_0",
            "q4_0_unrotated": "q4_0_unrotated",
        },
        ("fp16",),
    ),
    "--thinking-token-budget": values(
        "--thinking-token-budget", {"0": 0, "4096": 4096}, ("-1", "1.5", "many")
    ),
    "--thinking-end-text": values(
        "--thinking-end-text",
        {"</think>": "</think>", "Time is up.\n</think>": "Time is up.\n</think>"},
        (),
    ),
    "--state-format": values(
        "--state-format", {"fp32": "fp32", "bf16": "bf16"}, ("fp16", "int8")
    ),
    "--disable-ane": ([(["--disable-ane"], True)], []),
    "--ane-ffn-share": values(
        "--ane-ffn-share",
        {"auto": None, "0": 0.0, "1": 1.0, "0.5": 0.5},
        ("-1", "1.1", "nan", "inf", "half"),
    ),
    "--ane-ffn-minimum-rows": (
        [
            (["--ane-ffn-share=0.5", "--ane-ffn-minimum-rows=512"], 512),
            (["--ane-ffn-share=0.5", "--ane-ffn-minimum-rows=8192"], 8192),
        ],
        [
            ["--ane-ffn-minimum-rows=512"],
            ["--ane-ffn-minimum-rows=511"],
            ["--ane-ffn-minimum-rows=8193"],
        ],
    ),
    "--ane-power-recovery": values(
        "--ane-power-recovery", {"off": "off", "on": "on"}, ("1", "yes")
    ),
    "--ane-power-reference": values(
        "--ane-power-reference",
        {"off": "off", "6": "6.0", "auto": "auto"},
        ("0", "-1", "nan", "inf", "6junk"),
    ),
    "--ane-recovery-idle-watts": values(
        "--ane-recovery-idle-watts",
        {"0": 0.0, "0.02": 0.02},
        ("-1", "nan", "inf", "quiet"),
    ),
    "--ane-recovery-max-power-percent": (
        [
            (
                [
                    "--ane-power-reference=6",
                    "--ane-power-recovery=on",
                    "--ane-recovery-max-power-percent=0",
                ],
                0.0,
            ),
            (
                [
                    "--ane-power-reference=auto",
                    "--ane-power-recovery=on",
                    "--ane-recovery-max-power-percent=50",
                ],
                50.0,
            ),
        ],
        [
            ["--ane-recovery-max-power-percent=" + value]
            for value in ("-1", "101", "nan", "inf")
        ]
        + [["--ane-power-recovery=on", "--ane-recovery-max-power-percent=50"]],
    ),
    "--prompt-lookup": values(
        "--prompt-lookup", {"off": "off", "on": "on"}, ("1", "yes")
    ),
    "--prompt-lookup-min-match": values(
        "--prompt-lookup-min-match", {"2": 2, "4": 4, "3": 3}, ("1", "5", "3x")
    ),
    "--compact-draft-head": (
        [
            (["--compact-draft-head", "off"], "off"),
            (
                [
                    "--compact-draft-head",
                    "on",
                    "--compact-draft-ranking",
                    "/tmp/public-ranking",
                ],
                "on",
            ),
        ],
        [["--compact-draft-head", value] for value in ("1", "yes")],
    ),
    "--compact-draft-capacity": values(
        "--compact-draft-capacity",
        {"512": 512, "98304": 98304, "65536": 65536},
        ("256", "33000", "98305", "nan"),
    ),
    "--compact-draft-ranking": values(
        "--compact-draft-ranking", {"/tmp/rank file": Path("/tmp/rank file")}, ("",)
    ),
    "--fast-topk": values("--fast-topk", {"off": "off", "on": "on"}, ("1", "yes")),
    "--max-memory": values(
        "--max-memory",
        {
            "auto": None,
            "28G": 28 * 1024**3,
            "1GiB": 1024**3,
            "512mb": 512 * 1024**2,
            "1073741824": 1024**3,
        },
        ("0", "-1G", "bad", "G", str(2**64)),
    ),
    "--idle-release": values(
        "--idle-release",
        {"600": 600.0, "90s": 90.0, "30M": 1800.0, "1.5h": 5400.0, "off": math.inf},
        ("0", "-1m", "inf", "nan", "10x", "m", "never"),
    ),
    "--max-cache-disk": values(
        "--max-cache-disk",
        {"auto": None, "0": 0, "5G": 5 * 1024**3},
        ("-1", "0G", "5X", "nan"),
    ),
    "--max-cache-disk-writes": (
        [
            (["--max-cache-disk-writes=0"], 0),
            (["--max-cache-disk-writes=200G"], 200 * 1024**3),
        ],
        [
            ["--max-cache-disk-writes=200G", "--max-cache-disk=0"],
            ["--max-cache-disk-writes=-1"],
            ["--max-cache-disk-writes=5X"],
            ["--max-cache-disk-writes=auto"],
        ],
    ),
    "--persistent-cache": (
        [
            (["--persistent-cache"], True),
            (["--persistent-cache", "--max-cache-disk=5G"], True),
        ],
        [["--persistent-cache", "--max-cache-disk=0"]],
    ),
    "--cache-dir": (
        [
            (
                ["--cache-dir=/srv/cache", "--persistent-cache", "--max-cache-disk=5G"],
                Path("/srv/cache"),
            ),
            (
                ["--cache-dir=~/cache", "--persistent-cache", "--max-cache-disk=5G"],
                Path.home() / "cache",
            ),
        ],
        [
            ["--cache-dir=/srv/cache"],
            ["--cache-dir=", "--persistent-cache", "--max-cache-disk=5G"],
        ],
    ),
    "--max-context": values(
        "--max-context",
        {"auto": None, "100K": 102400, "256k": 262144, "262144": 262144, "1": 1},
        ("0", "-1", "257K", "262145", "bad"),
    ),
    "--decode-share": values(
        "--decode-share",
        {"0": 0.0, "0.25": 0.25, "2": 2.0},
        ("-0.5", "inf", "nan", "half"),
    ),
    "--allowed-host": values(
        "--allowed-host", {"proxy.example": "proxy.example"}, (), repeatable=True
    ),
    "--allowed-origin": values(
        "--allowed-origin",
        {
            "tauri://localhost": ("tauri", "localhost", None),
            "http://[::1]:3000": ("http", "::1", 3000),
            "*": ANY_ORIGIN,
        },
        ("http://localhost/app", "tauri://*", "localhost"),
        repeatable=True,
    ),
    "--max-request-size": values(
        "--max-request-size",
        {"256M": 256 * 1024**2, "1G": 1024**3},
        ("auto", "0", "-1G", "bad", str(2**64)),
    ),
    "--max-image-pixels": values(
        "--max-image-pixels",
        {str(images.MIN_PIXELS): images.MIN_PIXELS, "1048576": 1048576},
        (str(images.MIN_PIXELS - 1), str(images.MAX_PIXELS + 1), "-1", "many"),
    ),
    "--request-timeout": values(
        "--request-timeout",
        {"3600": 3600.0, "0.5": 0.5, "90s": 90.0, "30M": 1800.0, "1.5h": 5400.0},
        ("0", "-1", "-1m", "inf", "nan", "10x", "m", "off", "soon"),
    ),
    "--queue-size": values(
        "--queue-size", {"64": 64, "1": 1}, ("0", "-1", "1.5", "many")
    ),
    "--api-key": values(
        "--api-key",
        {"secret-key": "secret-key"},
        ("", "two words", "key\n", "非ASCII"),
    ),
    "--allow-idle-sleep": ([(["--allow-idle-sleep"], True)], []),
    "--no-webui": ([(["--no-webui"], True)], []),
}


def dest(flag):
    return flag.removeprefix("--").replace("-", "_")


def native_command_lines():
    """(name, serve options, serve-native arguments after the model directory)
    of each line of the command lines both sides read."""
    golden = api.ROOT / "dev/tests/engine/native_command_golden.txt"
    for line in golden.read_text().splitlines():
        if line and not line.startswith("#"):
            name, options, arguments = line.split("|")
            yield name.strip(), options.split(), arguments.split()


class ServeOptionsTests(unittest.TestCase):
    def setUp(self):
        # The parsers' defaults ignore the caller's Splash settings.
        self.enterContext(mock.patch.dict(os.environ))
        for name in (
            "SPLASH_API_KEY",
            "SPLASH_DEFAULT_REASONING_EFFORT",
            "SPLASH_THINKING_TOKEN_BUDGET",
            "SPLASH_THINKING_END_TEXT",
        ):
            os.environ.pop(name, None)

    def refuse(self, parse, arguments):
        with (
            mock.patch("sys.stderr", io.StringIO()) as stderr,
            self.assertRaises(SystemExit) as raised,
        ):
            parse(arguments)
        self.assertEqual(raised.exception.code, 2)
        return stderr.getvalue()

    def test_both_parsers_accept_and_reject_the_same_values(self):
        self.assertEqual(
            set(OPTIONS), {option.flag for option in serve_options.SERVE_OPTIONS}
        )
        defaults = [vars(parse(required)) for parse, required in PARSERS]
        for option in serve_options.SERVE_OPTIONS:
            self.assertEqual(
                defaults[0][option.dest], defaults[1][option.dest], option.flag
            )
        for flag, (accepted, refused) in OPTIONS.items():
            for (parse, required), (arguments, parsed) in (
                (parser, case) for parser in PARSERS for case in accepted
            ):
                with self.subTest(parser=parse.__module__, arguments=arguments):
                    self.assertEqual(
                        getattr(parse([*required, *arguments]), dest(flag)), parsed
                    )
            for (parse, required), arguments in (
                (parser, case) for parser in PARSERS for case in refused
            ):
                with self.subTest(parser=parse.__module__, arguments=arguments):
                    self.refuse(parse, [*required, *arguments])

    def test_both_parsers_check_defaults_from_the_environment(self):
        for name, value, parsed, flag in (
            (
                "SPLASH_DEFAULT_REASONING_EFFORT",
                "low",
                "low",
                "--default-reasoning-effort",
            ),
            ("SPLASH_THINKING_TOKEN_BUDGET", "4096", 4096, "--thinking-token-budget"),
            ("SPLASH_API_KEY", "environment-key", "environment-key", "--api-key"),
        ):
            for parse, required in PARSERS:
                with (
                    self.subTest(parser=parse.__module__, name=name),
                    mock.patch.dict(os.environ, {name: value}),
                ):
                    self.assertEqual(getattr(parse(required), dest(flag)), parsed)
                    # The command line outranks the environment.
                    explicit = OPTIONS[flag][0][0]
                    self.assertEqual(
                        getattr(parse([*required, *explicit[0]]), dest(flag)),
                        explicit[1],
                    )
                with (
                    self.subTest(parser=parse.__module__, name=name, valid=False),
                    mock.patch.dict(os.environ, {name: "two words"}),
                ):
                    self.assertIn(flag, self.refuse(parse, required))

    def test_performance_environment_does_not_override_startup_arguments(self):
        environment = {
            "SPLASH_ANE_POWER_RECOVERY": "1",
            "SPLASH_ANE_POWER_REFERENCE": "auto",
            "SPLASH_ANE_RECOVERY_IDLE_WATTS": "invalid",
            "SPLASH_ANE_RECOVERY_MAX_POWER_PERCENT": "invalid",
            "SPLASH_PROMPT_LOOKUP": "1",
            "SPLASH_COMPACT_DRAFT_HEAD": "1",
            "SPLASH_FAST_TOPK": "1",
            "SPLASH_PROMPT_LOOKUP_MIN_MATCH": "invalid",
            "SPLASH_COMPACT_DRAFT_CAPACITY": "invalid",
        }
        for parse, required in PARSERS:
            with mock.patch.dict(os.environ, environment):
                args = parse(required)
                self.assertEqual(args.ane_power_recovery, "off")
                self.assertEqual(args.ane_power_reference, "off")
                self.assertEqual(args.prompt_lookup, "off")
                self.assertEqual(args.compact_draft_head, "off")
                self.assertEqual(args.fast_topk, "off")
                self.assertEqual(args.prompt_lookup_min_match, 2)
                self.assertEqual(args.compact_draft_capacity, 98304)

    def test_compact_requires_an_explicit_file_and_file_alone_does_not_enable(self):
        for parse, required in PARSERS:
            self.assertIn(
                "--compact-draft-ranking",
                self.refuse(parse, [*required, "--compact-draft-head", "on"]),
            )
            args = parse(
                [
                    *required,
                    "--compact-draft-ranking",
                    "/tmp/missing-public-rank.splash",
                ]
            )
            self.assertEqual(args.compact_draft_head, "off")
            args = parse(
                [
                    *required,
                    "--compact-draft-head",
                    "on",
                    "--compact-draft-ranking",
                    "/tmp/public-rank.splash",
                ]
            )
            self.assertEqual(args.compact_draft_head, "on")

    def test_both_parsers_take_the_thinking_end_text_from_the_environment(self):
        # Any text parses; the server checks it with the model's tokenizer.
        text = "Time is up.\n</think>"
        for parse, required in PARSERS:
            with (
                self.subTest(parser=parse.__module__),
                mock.patch.dict(os.environ, {"SPLASH_THINKING_END_TEXT": text}),
            ):
                self.assertEqual(parse(required).thinking_end_text, text)
                # The command line outranks the environment.
                given = parse([*required, "--thinking-end-text=</think>"])
                self.assertEqual(given.thinking_end_text, "</think>")

    def test_each_parse_has_its_own_lists(self):
        for parse, required in PARSERS:
            for option in serve_options.SERVE_OPTIONS:
                if option.options.get("action") != "append":
                    continue
                with self.subTest(parser=parse.__module__, flag=option.flag):
                    getattr(parse(required), option.dest).append("changed")
                    self.assertEqual(getattr(parse(required), option.dest), [])

    def test_launcher_forwards_exactly_what_the_server_parses(self):
        launched = [
            *(
                argument
                for accepted, _ in OPTIONS.values()
                for arguments, _ in accepted[-1:]
                for argument in arguments
            ),
            # A repeatable option keeps every value, in order.
            "--served-model-name=stable",
            "--allowed-host=proxy.local",
            "--allowed-origin=tauri://localhost",
        ]
        args = launcher.parse_args([*LAUNCHER_ARGS, *launched])
        argv = serve_options.serve_argv(args)
        environment = serve_options.serve_environment(args)
        # Every option but the key travels on the command line; the key only
        # in the environment.
        self.assertEqual(
            {argument.partition("=")[0] for argument in argv},
            {
                option.flag
                for option in serve_options.SERVE_OPTIONS
                if not option.secret
            },
        )
        self.assertFalse(any("secret-key" in argument for argument in argv))
        self.assertEqual(environment, {"SPLASH_API_KEY": "secret-key"})
        with mock.patch.dict(os.environ, environment):
            served = api.parse_args([*SERVER_ARGS, *argv])
        for option in serve_options.SERVE_OPTIONS:
            self.assertEqual(
                getattr(served, option.dest), getattr(args, option.dest), option.flag
            )
        # Defaults stay with the server, which reads them itself.
        defaults = launcher.parse_args(LAUNCHER_ARGS)
        self.assertEqual(serve_options.serve_argv(defaults), [])
        self.assertEqual(serve_options.serve_environment(defaults), {})

    def test_serve_options_imports_only_the_standard_library(self):
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-c",
                "import sys; sys.path.insert(0, sys.argv[1]); "
                "import server.serve_options; "
                "print(sorted({name.partition('.')[0] for name in sys.modules} "
                "- set(sys.stdlib_module_names)))",
                str(ROOT),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "['__main__', 'server']")

    def test_server_requires_explicit_model_and_paths(self):
        model = "community/custom-splash"
        package = api.ROOT / "install/models" / model
        required = [
            str(package),
            "--tokenizer",
            str(package / "tokenizer"),
            "--model",
            model,
        ]
        for arguments in (
            [],
            required[:-2],
            [*required[:-1], "qwen3.8-27b"],
            [*required[:-1], "Qwen3.8-27B"],
            [*required[:-1], "https://huggingface.co/community/model"],
            [*required[:-1], "community/../model"],
        ):
            with (
                self.subTest(arguments=arguments),
                mock.patch("sys.stderr"),
                self.assertRaises(SystemExit),
            ):
                api.parse_args(arguments)
        args = api.parse_args(required)
        self.assertEqual(Path(args.model_root), package)
        self.assertEqual(Path(args.tokenizer), package / "tokenizer")
        self.assertIsNone(args.request_timeout)
        self.assertEqual(args.model, model)
        self.assertEqual(Path(args.binary).name, "splash")
        with mock.patch("sys.stderr"):
            for value in ("-1", "65536"):
                with self.assertRaises(SystemExit):
                    api.parse_args([*required, "--port", value])

    def test_native_command_passes_each_serve_option(self):
        model = "community/custom-splash"
        package = api.ROOT / "install/models" / model
        required = [
            str(package),
            "--tokenizer",
            str(package / "tokenizer"),
            "--model",
            model,
        ]
        args = api.parse_args(required)
        # The engine's command line for each option, which the engine reads as
        # the server means it (native_arguments_test.cpp). The engine keeps
        # its own default of an option the server does not pass, and the
        # per-image patch limit is the most patches a resized image within
        # the pixel cap can have, a whole number of merge units. A relaunch's
        # line follows a first engine that served 61433 tokens.
        for name, options, arguments in native_command_lines():
            with self.subTest(name=name):
                self.assertEqual(
                    api._native_command(
                        api.parse_args([*required, *options]),
                        61433 if name.startswith("relaunch") else None,
                    ),
                    [args.binary, "serve-native", args.model_root, *arguments],
                )

    def test_experimental_quantized_ffn_reaches_native_and_relaunch(self):
        for parse, required in PARSERS:
            self.assertEqual(parse(required).experimental_quantized_ffn, "off")
        for setting in ("off", "on"):
            args = api.parse_args(
                [*SERVER_ARGS, "--experimental-quantized-ffn", setting]
            )
            for context in (None, 61433):
                command = api._native_command(args, context)
                if setting == "on":
                    index = command.index("--experimental-quantized-ffn")
                    self.assertEqual(command[index + 1], "on")
                else:
                    self.assertNotIn("--experimental-quantized-ffn", command)
            for decision_type in ("laya", "vllm-d3"):
                args.decision_type = decision_type
                args.kv_format = "q4h"
                if setting == "on":
                    with self.assertRaisesRegex(ValueError, "only supports Qwen3.5-4B"):
                        api._native_command(args)
                else:
                    self.assertNotIn(
                        "--experimental-quantized-ffn", api._native_command(args)
                    )


if __name__ == "__main__":
    unittest.main()
