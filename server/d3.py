"""Noncausal Qwen D3 prompts and answer-code scores for System One."""

import json

from jinja2.sandbox import ImmutableSandboxedEnvironment

from . import judgments, protocol
from .kev import KevScorer
from .tokenization import tokenizable


class D3Scorer(KevScorer):
    def __init__(self, tokenizer, config):
        if config.get("type") != "vllm-d3" or config.get("head_size") != 256:
            raise ValueError("unsupported D3 configuration")
        self.codes = config["codes"]
        self.token_ids = config["token_ids"]
        if (
            not 2 <= len(self.codes) <= 255
            or len(self.codes) != len(self.token_ids)
            or len(set(self.token_ids)) != len(self.token_ids)
        ):
            raise ValueError("invalid D3 answer-code mapping")
        for code, token in zip(self.codes, self.token_ids):
            if tokenizer.encode(code, add_special_tokens=False) != [token]:
                raise ValueError("D3 answer-code tokenizer mismatch: " + code)
        self.temperatures = config["temperatures"]
        # Reuse Kev's temperature validation, with D3's own head and template.
        import math

        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
            for value in (self.temperatures[k] for k in ("choice", "score", "noul"))
        ):
            raise ValueError("invalid D3 temperature")
        environment = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True
        )
        # Match the model/HF JSON filter: keep Unicode, HTML text and insertion order.
        environment.filters["tojson"] = lambda value: json.dumps(
            value, ensure_ascii=False
        )
        self.template = environment.from_string(config["chat_template"])
        self.tokenizer = tokenizer

    def encode(self, state, spec, question=None):
        if len(spec.labels) > len(self.codes):
            raise ValueError("too many D3 options")
        # Match llama.cpp's JSON ordering; score levels keep their ordinal order.
        order = self.order(spec)
        descriptions = spec.descriptions
        if question is not None and spec.kind in ("choice", "noul"):
            criteria = question.get("criteria") or {}
            descriptions = tuple(criteria.get(label) for label in spec.labels)
        options = [
            {"key": spec.labels[i], "description": descriptions[i], "label": code}
            for code, i in zip(self.codes, order)
        ]
        prompt = self.template.render(
            state=state,
            instructions=question.get("instructions", spec.instructions)
            if question
            else spec.instructions,
            options=options,
            images=[],
            type=spec.kind,
        )
        return (
            self.tokenizer.encode(tokenizable(prompt), add_special_tokens=False),
            tuple(self.token_ids[: len(options)]),
        )

    def flags(self, spec):
        # The normal score protocol carries vocabulary IDs, not Kev marker positions.
        return protocol.RequestFlag(0)

    def answer(self, spec, probabilities):
        return judgments.systemone_answer(spec, probabilities)

    def status(self):
        return {
            "type": "vllm-d3",
            "architecture": "qwen35",
            "generation": False,
            "readout": "last token, selected vocabulary code rows",
            "answer_codes": len(self.codes),
            "attention": "noncausal full attention; causal GDN",
            "temperatures": dict(self.temperatures),
        }
