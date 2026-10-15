import copy
import tempfile
import unittest
from pathlib import Path

from tokenizers import pre_tokenizers

from dev.tests import fixture_files
from install import gguf


def fixture(hidden):
    julia = hidden == 384
    layers, vocab = (24, 256000) if julia else (30, 50368)
    special = (
        ["<pad>", "<eos>", "<bos>", "<unk>", "<mask>"]
        if julia
        else ["[PAD]", "[SEP]", "[CLS]", "[UNK]", "[MASK]"]
    )
    alphabet = (
        [f"<0x{i:02X}>" for i in range(256)] + ["▁"]
        if julia
        else sorted(pre_tokenizers.ByteLevel.alphabet())
    )
    tokens = special + alphabet
    tokens += [f"[unused_{i}]" for i in range(vocab - len(tokens))]
    return {
        "general.architecture": "modern-bert",
        "modern-bert.embedding_length": hidden,
        "modern-bert.block_count": layers,
        "modern-bert.vocab_size": vocab,
        "modern-bert.context_length": 8192,
        "modern-bert.attention.head_count": 6 if julia else 16,
        "modern-bert.attention.causal": False,
        "modern-bert.decision.type": "laya",
        "modern-bert.decision.block_count": 2,
        "modern-bert.decision.max_head_tokens": 256 if julia else 192,
        "modern-bert.feed_forward_length": ([1152] * 22 + [1536] * 2)
        if julia
        else ([2624] * 28 + [4096] * 2),
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "mmbert" if julia else "modern-bert",
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.token_type": [3] * 5
        + [1] * len(alphabet)
        + [5] * (vocab - 5 - len(alphabet)),
        "tokenizer.ggml.merges": [],
        "tokenizer.ggml.bos_token_id": 2,
        "tokenizer.ggml.eos_token_id": 1,
        "tokenizer.ggml.padding_token_id": 0,
        "tokenizer.ggml.mask_token_id": 4,
        "tokenizer.ggml.seperator_token_id": 1,
        "tokenizer.ggml.add_bos_token": True,
        "tokenizer.ggml.add_eos_token": True,
        "tokenizer.chat_template.systemone": special[2]
        + "{{ type }} question: {{ instructions }}"
        + special[1]
        + "{% for o in options %}"
        + special[4]
        + " {{ o.key }}{% endfor %}"
        + special[1]
        + "{{ state }}"
        + special[1],
    }


class ModernBertMetadataTests(unittest.TestCase):
    def metadata(self, hidden):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        values = fixture(hidden)
        first = fixture_files.write_gguf(Path(directory.name) / "header.gguf", values)
        schema = gguf.modernbert_tensors(gguf.Metadata(first))
        types = {"Q8_0": 8, "F32": 0, "BF16": 30}
        tensors = [
            (name, shape, types[next(iter(kinds))], b"")
            for name, (shape, kinds) in schema.items()
        ]
        return gguf.Metadata(
            fixture_files.write_gguf(
                Path(directory.name) / "model.gguf", values, tensors
            ),
            tensors=True,
        )

    def test_both_geometries_and_tokenizers_are_supported_without_local_weights(self):
        for hidden, layers in ((384, 24), (1024, 30)):
            with self.subTest(hidden=hidden):
                metadata = self.metadata(hidden)
                gguf.require_loadable(metadata)
                config = gguf.model_config(metadata)
                self.assertEqual(config["text_config"]["hidden_size"], hidden)
                self.assertEqual(config["text_config"]["num_hidden_layers"], layers)
                self.assertEqual(config["decision_config"]["type"], "laya")
                self.assertIn(
                    "tokenizer/tokenizer.json", gguf.tokenizer_files(metadata)
                )

    def test_bad_shape_type_missing_head_and_causal_mode_are_rejected(self):
        metadata = self.metadata(384)
        for kind in ("shape", "type", "missing", "causal"):
            with self.subTest(kind=kind):
                bad = copy.copy(metadata)
                bad.shapes, bad.tensors, bad.values = (
                    dict(metadata.shapes),
                    dict(metadata.tensors),
                    dict(metadata.values),
                )
                if kind == "shape":
                    bad.shapes["cls.output.weight"] = (384, 512)
                if kind == "type":
                    bad.tensors["cls.output.weight"] = 8
                if kind == "missing":
                    del bad.shapes["cls.output.bias"]
                if kind == "causal":
                    bad.values["modern-bert.attention.causal"] = True
                with self.assertRaises(gguf.ModelError):
                    gguf.require_loadable(bad)
