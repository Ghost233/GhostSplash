"""H2O-Lightning text decisions with the GGUF System One prompt contract."""

import json
import re

from .d3 import D3Scorer
from .kev import KevScorer
from .tokenization import tokenizable


class H2OScorer(D3Scorer):
    def __init__(self, tokenizer, config):
        super().__init__(tokenizer, config)
        self.causal = config.get("causal", True)
        spellings = set(getattr(tokenizer, "all_special_tokens", ()))
        if hasattr(tokenizer, "get_added_vocab"):
            spellings.update(tokenizer.get_added_vocab())
        self.special = (
            re.compile(
                "|".join(re.escape(s) for s in sorted(spellings, key=len, reverse=True))
            )
            if spellings
            else None
        )

    def quote(self, text):
        if self.special is None:
            return text
        return self.special.sub(
            lambda match: match[0].replace("<", r"\u003c").replace(">", r"\u003e"), text
        )

    @staticmethod
    def order(spec):
        if spec.kind == "noul":
            return tuple(spec.labels.index(label) for label in ("true", "false"))
        return D3Scorer.order(spec)

    def encode(self, state, spec, question=None):
        if len(spec.labels) > len(self.codes):
            raise ValueError("too many H2O options")
        question = question or {}
        instructions = question.get("instructions", "")
        instructions = (
            instructions
            if isinstance(instructions, str)
            else json.dumps(instructions, indent=1, sort_keys=True)
        )
        state = (
            state
            if isinstance(state, str)
            else json.dumps(
                state, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            )
        )
        criteria = question.get("criteria") or {}
        options = []
        for code, index in zip(self.codes, self.order(spec)):
            key = spec.labels[index]
            description = (
                spec.descriptions[index] if spec.kind == "score" else criteria.get(key)
            )
            if spec.kind == "noul" and description is None:
                description = criteria.get("yes" if key == "true" else "no")
            if description is not None and not isinstance(description, str):
                description = json.dumps(
                    description, ensure_ascii=False, sort_keys=True
                )
            options.append(
                {
                    "key": self.quote(key),
                    "label": code,
                    "description": self.quote(description)
                    if description is not None
                    else None,
                }
            )
        prompt = self.template.render(
            state=self.quote(state),
            instructions=self.quote(instructions.lstrip()),
            options=options,
            images=[],
            type=spec.kind,
        )
        return self.tokenizer.encode(
            tokenizable(prompt), add_special_tokens=False
        ), tuple(self.token_ids[: len(options)])

    def status(self):
        return super().status() | {
            "attention": "causal full attention and GDN"
            if self.causal
            else "noncausal full attention; causal GDN"
        }

    def answer(self, spec, probabilities):
        return KevScorer.answer(self, spec, probabilities)
