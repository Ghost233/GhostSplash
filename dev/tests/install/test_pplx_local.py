import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from dev.tests.installer_fixtures import GGML, fixture, write_target
from install.local import prepare_local_model


def pplx_target(path):
    values = {
        key.replace("qwen35moe.", "qwen35."): value
        for key, value in fixture(native=True).items()
        if not key.startswith("qwen35moe.expert")
    }
    values.update(
        {
            "general.architecture": "qwen35",
            "qwen35.embedding_length": 2560,
            "qwen35.block_count": 32,
            "qwen35.nextn_predict_layers": 0,
            "qwen35.feed_forward_length": 9216,
            "qwen35.ssm.inner_size": 4096,
            "qwen35.ssm.time_step_rank": 32,
            "qwen35.attention.head_count": 16,
            "qwen35.attention.head_count_kv": 4,
            "qwen35.rope.dimension_count": 64,
            "qwen35.attention.causal": False,
            "pplx_decider.protocol_version": 3,
            "pplx_decider.template": "joint-question-inline-readout-v1",
            "pplx_decider.marker": "<|fim_suffix|>",
            "pplx_decider.temperature": 1.5539092221462472,
            "pplx_decider.codes": ["A", "B"],
            "pplx_decider.token_ids": [
                values["tokenizer.ggml.tokens"].index("A"),
                values["tokenizer.ggml.tokens"].index("B"),
            ],
            "pplx_decider.max_pixels": 1638400,
            "pplx_decider.escape_tokens": ["<|fim_suffix|>"],
        }
    )
    values["tokenizer.ggml.tokens"][3000] = "<|fim_suffix|>"
    tensors = [
        ("output_norm.weight", [2560], GGML["F32"]),
        ("token_embd.weight", [2560, 248320], GGML["Q4_K"]),
        ("output.weight", [2560, 248320], GGML["Q4_K"]),
    ]
    common = {
        "attn_norm.weight": ([2560], "F32"),
        "post_attention_norm.weight": ([2560], "F32"),
        "ffn_down.weight": ([9216, 2560], "Q4_K"),
        "ffn_gate.weight": ([2560, 9216], "Q4_K"),
        "ffn_up.weight": ([2560, 9216], "Q4_K"),
    }
    attention = {
        "attn_q.weight": ([2560, 8192], "Q4_K"),
        "attn_k.weight": ([2560, 1024], "Q4_K"),
        "attn_v.weight": ([2560, 1024], "Q4_K"),
        "attn_q_norm.weight": ([256], "F32"),
        "attn_k_norm.weight": ([256], "F32"),
        "attn_output.weight": ([4096, 2560], "Q4_K"),
    }
    recurrent = {
        "attn_qkv.weight": ([2560, 8192], "Q4_K"),
        "attn_gate.weight": ([2560, 4096], "Q4_K"),
        "ssm_beta.weight": ([2560, 32], "F16"),
        "ssm_alpha.weight": ([2560, 32], "F16"),
        "ssm_conv1d.weight": ([4, 8192], "F32"),
        "ssm_a": ([32], "F32"),
        "ssm_dt.bias": ([32], "F32"),
        "ssm_norm.weight": ([128], "F32"),
        "ssm_out.weight": ([4096, 2560], "Q4_K"),
    }
    for layer in range(32):
        weights = common | (attention if layer % 4 == 3 else recurrent)
        tensors.extend(
            (f"blk.{layer}.{name}", shape, GGML[kind])
            for name, (shape, kind) in weights.items()
        )
    return write_target(path, values, tensors)


class PplxLoadingTests(unittest.TestCase):
    def test_model_loading_preserves_the_v3_readout_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = pplx_target(root / "pplx.gguf")
            with contextlib.redirect_stdout(io.StringIO()):
                prepared = prepare_local_model(
                    target, language_only=True, models_dir=root / "models"
                )
            config = json.loads((prepared / "config.json").read_text())[
                "decision_config"
            ]
            self.assertEqual(config["type"], "pplx-decider-v3")
            self.assertEqual(config["codes"], ["A", "B"])
            self.assertAlmostEqual(config["temperature"], 1.5539092221462472)


if __name__ == "__main__":
    unittest.main()
