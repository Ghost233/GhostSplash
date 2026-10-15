"""ModernBERT decision prompts, option budgets and calibrated readout."""

import json
import math

from jinja2.sandbox import ImmutableSandboxedEnvironment

from . import protocol as wire
from .kev import KevScorer


class ModernBertScorer(KevScorer):
    def __init__(self, tokenizer, config):
        if config.get("type") != "laya" or config.get("head_size") != 1:
            raise ValueError("unsupported encoder decision configuration")
        self.temperatures = dict(config["temperatures"])
        if any(
            isinstance(v, bool)
            or not isinstance(v, (float, int))
            or not math.isfinite(v)
            or v <= 0
            for v in self.temperatures.values()
        ):
            raise ValueError("invalid encoder decision temperature")
        self.max_head_tokens = config["max_head_tokens"]
        self.mask = config["mask_token_id"]
        self.sep = config["sep_token_id"]
        self.tokenizer = tokenizer
        self.marker = tokenizer.convert_ids_to_tokens(self.mask)
        if tokenizer.encode(self.marker, add_special_tokens=False) != [self.mask]:
            raise ValueError("encoder tokenizer marker mismatch")
        environment = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True
        )
        environment.filters["tojson"] = lambda value, **kwargs: json.dumps(
            value, ensure_ascii=False, **kwargs
        )
        self.template = environment.from_string(config["chat_template"])

    @staticmethod
    def order(spec):
        # ModernBERT reference rendering preserves the caller's option order.
        return tuple(range(len(spec.labels)))

    def _escape(self, value):
        if isinstance(value, str):
            return value.replace(self.marker, " ")
        if isinstance(value, dict):
            return {key: self._escape(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._escape(item) for item in value]
        return value

    def encode(self, state, spec, question=None):
        criteria = question.get("criteria") if question else None
        descriptions = spec.descriptions
        if question is not None and spec.kind in ("choice", "noul"):
            descriptions = tuple((criteria or {}).get(label) for label in spec.labels)
        elif isinstance(criteria, list) and spec.kind == "score":
            descriptions = tuple(criteria)
        options = [
            {"key": spec.labels[i], "description": descriptions[i]}
            for i in self.order(spec)
        ]
        prompt = self.template.render(
            state=self._escape(state),
            type=spec.kind,
            instructions=self._escape(
                question.get("instructions", spec.instructions)
                if question
                else spec.instructions
            ),
            options=self._escape(options),
        )
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        markers = [i for i, token in enumerate(tokens) if token == self.mask]
        if (
            len(markers) != len(options)
            or not markers
            or markers[0] < 2
            or tokens[markers[0] - 1] != self.sep
            or tokens[-1] != self.sep
        ):
            raise ValueError("unexpected layout of the encoder decision prompt")
        head_end = markers[0] - 1
        end = tokens.index(self.sep, markers[-1])
        spans = [
            tokens[start : (markers[i + 1] if i + 1 < len(markers) else end)][:49]
            for i, start in enumerate(markers)
        ]
        count = sum(map(len, spans))
        if count + 16 > self.max_head_tokens:
            limit = max(
                4,
                (self.max_head_tokens - min(self.max_head_tokens, 16)) // len(options),
            )
            spans = [span[:limit] for span in spans]
            count = sum(map(len, spans))
        question_limit = max(8, self.max_head_tokens - min(self.max_head_tokens, count))
        output = [tokens[0], *tokens[1 : min(head_end, 1 + question_limit)], self.sep]
        positions = []
        for span in spans:
            positions.append(len(output))
            output.extend(span)
        output.extend(tokens[end:])
        return output, tuple(positions)

    def temperature(self, spec):
        n = len(spec.labels)
        bucket = "2" if n <= 2 else "3_5" if n <= 5 else "6_10" if n <= 10 else "11"
        return self.temperatures.get(
            f"{spec.kind}.{bucket}", self.temperatures.get(spec.kind, 1.0)
        )

    def flags(self, spec):
        kind = {
            "choice": wire.RequestFlag(0),
            "score": wire.RequestFlag.DECISION_SCORE_TYPE,
            "noul": wire.RequestFlag.DECISION_NOUL_TYPE,
        }[spec.kind]
        return wire.RequestFlag.DECISION_SCORING | kind

    def status(self):
        return {
            "type": "laya",
            "architecture": "modern-bert",
            "head_size": 1,
            "generation": False,
            "readout": "typed option-marker scores",
            "max_head_tokens": self.max_head_tokens,
            "temperatures": dict(self.temperatures),
        }
