import contextlib
import io
import json
import struct
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from dev.tools.compact_draft_ranking import (
    corpus_counts,
    corpus_coverage,
    main,
    rank_corpus,
    write_ranking,
)


class CompactRankingTest(unittest.TestCase):
    def test_public_binary_and_no_overwrite(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "ranking"
            write_ranking(path, [2, 0, 1])
            self.assertEqual(
                path.read_bytes(),
                b"SPLCHD02" + struct.pack("<4I", 3, 2, 0, 1),
            )
            with self.assertRaises(FileExistsError):
                write_ranking(path, [0, 1, 2])

    def test_legacy_import_keeps_token_order_and_leaves_source_intact(self):
        with TemporaryDirectory() as directory:
            source, output = Path(directory) / "old", Path(directory) / "public"
            original = b"SPLCHD01" + b"a" * 64 + struct.pack("<4I", 3, 2, 0, 1)
            source.write_bytes(original)
            argv = [
                "compact-ranking",
                "--import-legacy",
                str(source),
                "--output",
                str(output),
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                main()
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(
                output.read_bytes(), b"SPLCHD02" + struct.pack("<4I", 3, 2, 0, 1)
            )

    def test_invalid_permutation(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "ranking"
            for tokens in ([0, 0], [1, 2], [False, 1], [], [-1, 0]):
                with self.subTest(tokens=tokens):
                    with self.assertRaises(ValueError):
                        write_ranking(path, tokens)
                    self.assertFalse(path.exists())


class CorpusRankingTest(unittest.TestCase):
    def tokenizer(self):
        from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers

        codec = Tokenizer(
            models.WordLevel({"[UNK]": 0, "a": 1, "b": 2, "z": 3}, unk_token="[UNK]")
        )
        codec.pre_tokenizer = pre_tokenizers.Whitespace()
        codec.add_special_tokens([AddedToken("<eos>", special=True)])
        return codec

    def test_frequency_order_special_tokens_and_deterministic_ties(self):
        with TemporaryDirectory() as directory:
            first, second = Path(directory) / "first", Path(directory) / "second"
            first.write_text("b b a\n")
            second.write_text("z b\n")
            codec = self.tokenizer()
            order, counts = rank_corpus(codec, [first, second])
            self.assertEqual(order, [4, 2, 1, 3, 0])
            self.assertEqual(dict(counts), {2: 3, 1: 1, 3: 1})
            self.assertEqual(rank_corpus(codec, [second, first]), (order, counts))
            self.assertEqual(sorted(order), list(range(codec.get_vocab_size())))

    def test_empty_corpus_is_rejected_before_a_ranking_can_be_written(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "empty"
            path.write_text("")
            with self.assertRaisesRegex(ValueError, "no tokens"):
                rank_corpus(self.tokenizer(), [path])

    def test_validation_counts_use_the_same_tokenizer(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "heldout"
            path.write_text("z z b a\n")
            self.assertEqual(
                dict(corpus_counts(self.tokenizer(), [path])), {3: 2, 2: 1, 1: 1}
            )

    def test_coverage_reports_tokens_outside_the_fixed_rows(self):
        report = corpus_coverage({1: 1, 2: 1, 3: 2}, [4, 2, 1, 3, 0], 3)
        self.assertEqual(report["token_coverage_percent"], 50)
        self.assertEqual(report["outside_fixed_tokens"], 2)
        self.assertEqual(report["outside_fixed_token_ids"], 1)

    def test_corpus_cli_writes_public_ranking_and_reports_validation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.tokenizer().save(str(root / "tokenizer.json"))
            train, heldout, output = (
                root / "train.txt",
                root / "heldout.txt",
                root / "rank",
            )
            train.write_text("b b a\n")
            heldout.write_text("z z b a\n")
            argv = [
                "compact-ranking",
                "--tokenizer",
                str(root),
                "--corpus",
                str(train),
                "--validation-corpus",
                str(heldout),
                "--output",
                str(output),
            ]
            stdout = io.StringIO()
            with (
                mock.patch.object(sys, "argv", argv),
                contextlib.redirect_stdout(stdout),
            ):
                main()
            report = json.loads(stdout.getvalue())
            self.assertEqual(report["vocabulary_size"], 5)
            self.assertEqual(report["training"]["tokens"], 3)
            self.assertEqual(report["validation"]["tokens"], 4)
            self.assertEqual(report["special_tokens_reserved"], 1)
            self.assertEqual(output.read_bytes()[:8], b"SPLCHD02")
            self.assertEqual(
                struct.unpack("<6I", output.read_bytes()[8:]), (5, 4, 2, 1, 0, 3)
            )

    def test_corpus_cli_requires_tokenizer(self):
        argv = [
            "compact-ranking",
            "--corpus",
            "corpus.txt",
            "--output",
            "rank",
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as error:
                main()
        self.assertEqual(error.exception.code, 2)
