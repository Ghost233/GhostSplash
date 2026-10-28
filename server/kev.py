"""Kev GGUF pointer-head prompts and calibrated option probabilities."""

import json
import math
import re

from jinja2.sandbox import ImmutableSandboxedEnvironment

from . import judgments
from .tokenization import tokenizable

_SPECIAL = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def render_value(value, indent=0):
    pad = "  " * indent
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(
            pad + "- " + render_value(item, indent + 1).lstrip() for item in value
        )
    if isinstance(value, dict):
        return "\n".join(
            pad
            + key
            + (":\n" if isinstance(item, (dict, list)) else ": ")
            + render_value(item, indent + 1 if isinstance(item, (dict, list)) else 0)
            for key, item in sorted(value.items())
        )
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def text_value(value):
    # User text may not introduce extra option/query marker tokens.
    return _SPECIAL.sub(r"<¦\1¦>", render_value(value))


class KevScorer:
    def __init__(self, tokenizer, config):
        if config.get("type") != "kev" or config.get("head_size") != 512:
            raise ValueError("unsupported decision model configuration")
        self.temperatures = dict(config["temperatures"])
        for kind in ("choice", "score", "noul"):
            value = self.temperatures[kind]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError("invalid Kev decision temperature")
        template = config["chat_template"]
        if not isinstance(template, str) or not template.strip():
            raise ValueError("Kev systemone template is empty")
        self.template = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True
        ).from_string(template)
        self.tokenizer = tokenizer
        for token, token_id in (("<|box_end|>", 248050), ("<|fim_suffix|>", 248062)):
            if tokenizer.encode(token, add_special_tokens=False) != [token_id]:
                raise ValueError("Kev tokenizer marker mismatch: " + token)

    @staticmethod
    def order(spec):
        # Match the canonical GGUF System One reference: object keys are
        # sorted, rating levels stay ordered, and Noul presents no then yes.
        return (
            tuple(range(len(spec.labels)))
            if spec.kind == "score"
            else tuple(
                sorted(range(len(spec.labels)), key=lambda index: spec.labels[index])
            )
        )

    def encode_prefix(self, state):
        prompt = self.template.render(
            state=text_value(state), instructions="", options=[], type="choice"
        )
        prefix, marker, _ = prompt.partition("<|fim_middle|>")
        if not marker:
            raise ValueError("unexpected layout of the Kev state prefix")
        return self.tokenizer.encode(tokenizable(prefix), add_special_tokens=False)

    def encode(self, state, spec, question=None):
        descriptions = spec.descriptions
        if question is not None and spec.kind in ("choice", "noul"):
            criteria = question.get("criteria") or {}
            descriptions = tuple(criteria.get(label) for label in spec.labels)
        options = [
            {
                "key": text_value(spec.labels[index]),
                "description": text_value(descriptions[index]),
            }
            for index in self.order(spec)
        ]
        prompt = self.template.render(
            state=text_value(state),
            instructions=text_value(spec.instructions),
            options=options,
            type=spec.kind,
        )
        tokens = self.tokenizer.encode(tokenizable(prompt), add_special_tokens=False)
        positions = tuple(
            index for index, token in enumerate(tokens) if token == 248050
        )
        if len(positions) != len(options) or not tokens or tokens[-1] != 248062:
            raise ValueError("unexpected layout of the Kev decision prompt")
        return tokens, positions

    def temperature(self, spec):
        return self.temperatures[spec.kind]

    def flags(self, spec):
        from . import protocol as wire

        return wire.RequestFlag.DECISION_SCORING

    def probabilities(self, spec, logits):
        if len(logits) != len(spec.labels):
            raise ValueError("Kev result does not match the option count")
        probabilities = judgments.softmax(
            [value / self.temperature(spec) for value in logits]
        )
        ordered = [0.0] * len(probabilities)
        for index, probability in zip(self.order(spec), probabilities):
            ordered[index] = probability
        return ordered

    def answer(self, spec, probabilities):
        answer = judgments.systemone_answer(spec, probabilities)
        if spec.kind == "noul" or len(probabilities) < 2:
            return answer
        count = len(probabilities)
        if spec.kind == "choice":
            best = max(self.order(spec), key=probabilities.__getitem__)
            answer["choice"] = spec.labels[best]
            uniform = 1.0 / count
            answer["confidence"] = max(
                0.0, (probabilities[best] - uniform) / (1.0 - uniform)
            )
        else:
            mode = max(range(count), key=probabilities.__getitem__)
            distance = sum(
                p * abs(index - mode) for index, p in enumerate(probabilities)
            )
            uniform_distance = sum(
                abs(index - (count - 1) / 2) / count for index in range(count)
            )
            answer["confidence"] = max(0.0, 1.0 - distance / uniform_distance)
        return answer

    def status(self):
        return {
            "type": "kev",
            "head_size": 512,
            "generation": False,
            "readout": "option-end keys against final-position query",
            "temperatures": dict(self.temperatures),
        }
