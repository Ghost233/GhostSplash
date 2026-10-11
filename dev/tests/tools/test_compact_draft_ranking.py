import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from dev.tools.compact_draft_ranking import write_ranking


class CompactRankingTest(unittest.TestCase):
    def test_model_bound_binary_and_no_overwrite(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "ranking"
            write_ranking(path, "a" * 64, [2, 0, 1])
            self.assertEqual(
                path.read_bytes(),
                b"SPLCHD01" + b"a" * 64 + struct.pack("<4I", 3, 2, 0, 1),
            )
            with self.assertRaises(FileExistsError):
                write_ranking(path, "a" * 64, [0, 1, 2])

    def test_invalid_identity_and_permutation(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "ranking"
            for identity, tokens in [
                ("x" * 64, [0, 1]),
                ("a" * 64, [0, 0]),
                ("a" * 64, [1, 2]),
                ("a" * 64, [False, 1]),
                ("a" * 64, []),
            ]:
                with self.subTest(identity=identity, tokens=tokens):
                    with self.assertRaises(ValueError):
                        write_ranking(path, identity, tokens)
                    self.assertFalse(path.exists())
