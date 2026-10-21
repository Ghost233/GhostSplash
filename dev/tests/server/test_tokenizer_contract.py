import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers
from transformers.utils import logging as transformers_logging

from dev.tests.server_fixtures import byte_alphabet, main_args, unprepared_hold
from install.gguf import QWEN35_PATTERN
from server import constraints as generation_constraints
from server import server


class TokenizerContractTests(unittest.TestCase):
    def tokenizer(self):
        vocabulary = {
            "text": 0,
            "<|endoftext|>": 248044,
            "<|im_end|>": 248046,
            "</think>": 248069,
            "<tool_call>": 248058,
        }
        return mock.Mock(
            get_vocab=mock.Mock(return_value=vocabulary),
            encode=mock.Mock(side_effect=lambda text, **_: [vocabulary[text]]),
            eos_token_id=248046,
        )

    def test_padded_model_vocabulary_need_not_have_a_token_at_every_index(self):
        generation_constraints.validate_tokenizer(self.tokenizer())

    def test_mismatched_stop_or_thinking_tokens_are_rejected(self):
        for token in ("<|endoftext|>", "<|im_end|>", "</think>", "<tool_call>"):
            with self.subTest(token=token):
                tokenizer = self.tokenizer()
                tokenizer.get_vocab()[token] += 1
                with self.assertRaisesRegex(
                    server.engine_runtime.EngineUnhealthy, "native token"
                ):
                    generation_constraints.validate_tokenizer(tokenizer)

    def test_normalizer_cannot_split_a_native_special_token(self):
        tokenizer = self.tokenizer()
        tokenizer.encode.return_value = [1, 2]
        tokenizer.encode.side_effect = None
        with self.assertRaisesRegex(
            server.engine_runtime.EngineUnhealthy, "native token"
        ):
            generation_constraints.validate_tokenizer(tokenizer)

    def test_out_of_range_token_is_rejected(self):
        for token_id in (-1, 248320, True):
            with self.subTest(token_id=token_id):
                tokenizer = self.tokenizer()
                tokenizer.get_vocab()["community-token"] = token_id
                with self.assertRaisesRegex(
                    server.engine_runtime.EngineUnhealthy, "vocabulary"
                ):
                    generation_constraints.validate_tokenizer(tokenizer)

    def test_eos_configuration_must_match_a_native_stop_token(self):
        tokenizer = self.tokenizer()
        tokenizer.eos_token_id = 0
        with self.assertRaisesRegex(server.engine_runtime.EngineUnhealthy, "EOS token"):
            generation_constraints.validate_tokenizer(tokenizer)

    def test_startup_rejects_tokenizer_before_starting_the_native_worker(self):
        args = server.parse_args(["model", "--model", "owner/model"])
        tokenizer = self.tokenizer()
        tokenizer.get_vocab()["</think>"] = 0
        http = mock.Mock()
        with (
            mock.patch.object(server, "parse_args", return_value=args),
            unprepared_hold(),
            mock.patch.object(server.signal, "signal"),
            mock.patch.object(server, "load_thinking_key", return_value=None),
            mock.patch.object(
                server.AutoTokenizer, "from_pretrained", return_value=tokenizer
            ),
            mock.patch.object(server, "FrontendServer", return_value=http),
            mock.patch.object(server.engine_runtime, "MultiplexedRuntime") as native,
            mock.patch("sys.stderr", new_callable=io.StringIO) as stderr,
            mock.patch("sys.stdout", new_callable=io.StringIO),
            self.assertRaises(SystemExit) as raised,
        ):
            server.main()
        self.assertEqual(raised.exception.code, 1)
        self.assertIn("tokenizer must encode", stderr.getvalue())
        native.assert_not_called()
        http.server_activate.assert_not_called()
        http.server_close.assert_called_once()


class TokenizerLoadingTests(unittest.TestCase):
    def fixture(self, directory, model_type):
        vocabulary = {
            char: index for index, char in enumerate(byte_alphabet().values())
        }
        merges = [("P", "h"), ("Ph", "o"), ("Pho", "n"), ("Phon", "e"), ("i", "Phone")]
        for left, right in merges:
            vocabulary[left + right] = len(vocabulary)
        # Transformers only runs its Mistral heuristic above 100,000 tokens.
        while len(vocabulary) <= 100000:
            vocabulary[f"<unused_{len(vocabulary)}>"] = len(vocabulary)
        codec = Tokenizer(models.BPE(vocabulary, merges))
        codec.normalizer = normalizers.NFC()
        codec.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(Regex(QWEN35_PATTERN), behavior="isolated"),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        )
        codec.decoder = decoders.ByteLevel()
        codec.save(str(directory / "tokenizer.json"))
        (directory / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "TokenizersBackend"})
        )
        if model_type is not None:
            (directory / "config.json").write_text(
                json.dumps({"model_type": model_type})
            )
        return codec

    def load_at_startup(self, directory):
        captured = []

        def capture(tokenizer):
            captured.append(tokenizer)
            # Stop before native startup; this fixture tests the real CPU loader.
            raise KeyboardInterrupt

        logger = transformers_logging.get_logger(
            "transformers.tokenization_utils_tokenizers"
        )
        with (
            mock.patch.object(
                server,
                "parse_args",
                return_value=main_args(model_root=str(directory.parent)),
            ),
            mock.patch.object(
                server.assembly, "hold", return_value=(directory.parent, object())
            ),
            mock.patch.object(server, "_raise_descriptor_limit"),
            mock.patch.object(server.signal, "signal"),
            mock.patch.object(server.signal, "pthread_sigmask"),
            mock.patch.object(server, "load_thinking_key", return_value=None),
            mock.patch.object(server, "FrontendServer"),
            mock.patch.object(server, "print_status"),
            mock.patch.object(server, "validate_tokenizer", side_effect=capture),
            mock.patch.object(server.engine_runtime, "MultiplexedRuntime") as native,
            mock.patch.object(logger, "warning") as warning,
        ):
            server.main()
        native.assert_not_called()
        self.assertEqual(len(captured), 1)
        return captured[0], warning

    def test_qwen_without_transformers_version_preserves_tokens_without_mistral_warning(
        self,
    ):
        for model_type in (
            "qwen3_5",
            "qwen3_5_text",
            "qwen3_5_moe",
            "qwen3_5_moe_text",
        ):
            with self.subTest(model_type=model_type), TemporaryDirectory() as directory:
                root = Path(directory) / "tokenizer"
                root.mkdir()
                original = self.fixture(root, model_type)
                loaded, warning = self.load_at_startup(root)
                warning.assert_not_called()
                self.assertEqual(
                    loaded.backend_tokenizer.pre_tokenizer.__getstate__(),
                    original.pre_tokenizer.__getstate__(),
                )
                self.assertEqual(len(original.encode("iPhone").ids), 1)
                for text in ("iPhone", "中文与 eBay 123", "a\u0301\nfloat x = 0.5;"):
                    self.assertEqual(
                        loaded.encode(text, add_special_tokens=False),
                        original.encode(text, add_special_tokens=False).ids,
                    )

    def test_other_model_types_keep_transformers_detection(self):
        for model_type in ("mistral", "custom", None):
            with self.subTest(model_type=model_type), TemporaryDirectory() as directory:
                root = Path(directory) / "tokenizer"
                root.mkdir()
                original = self.fixture(root, model_type)
                loaded, warning = self.load_at_startup(root)
                if model_type is None:
                    warning.assert_not_called()
                else:
                    warning.assert_called_once()
                    self.assertIn("incorrect regex pattern", warning.call_args.args[0])
                self.assertEqual(
                    loaded.backend_tokenizer.pre_tokenizer.__getstate__(),
                    original.pre_tokenizer.__getstate__(),
                )


if __name__ == "__main__":
    unittest.main()
