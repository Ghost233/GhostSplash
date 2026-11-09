"""StartLux AutoThink text decision prompts and calibrated readouts.

Protocol reference: StartLuxLabs/StartLux-Decision, commit
a99977d9332decc9fb27a2488b4a83b533b6cd5f (Apache-2.0).
"""

import hashlib
import json
import re
import time
from dataclasses import dataclass, replace

from . import protocol
from .dialects import scoring
from .errors import ContextLengthError, RequestValidationError, field_error
from .kev import KevScorer
from .tokenization import tokenizable

SYSTEM = "Apply the criterion to the evidence. Choose exactly one listed option. Answer with its letter only."
PREFIX = "Let me work through this carefully, step by step.\n"
CLOSE = "\n</think>\n\n"
CUT = "\n\nI have to decide now with what I have worked out so far.\n</think>\n\n"
IMAGE_SLOT = "\0IMAGE\0"


def _annotate(value):
    if isinstance(value, list):
        if len(value) >= 8:
            return [
                (
                    {"_index": i, **_annotate(item)}
                    if isinstance(item, dict)
                    else {"_index": i, "value": _annotate(item)}
                )
                for i, item in enumerate(value)
            ]
        return [_annotate(item) for item in value]
    if isinstance(value, dict):
        return {key: _annotate(item) for key, item in value.items()}
    return value


def _text(value):
    if value is None:
        return ""
    return (
        value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    ).strip()


class StartLuxScorer:
    kind = "startlux-autothink"

    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.token_ids = config["token_ids"]
        self.temperatures = config["temperatures"]
        if not 2 <= len(self.token_ids) <= 26:
            raise ValueError("invalid StartLux letter tokens")
        for i, token in enumerate(self.token_ids):
            if tokenizer.encode(chr(65 + i), add_special_tokens=False) != [token]:
                raise ValueError("StartLux letter-token mismatch")

    def messages(self, state, spec, question, *, image_count=0):
        evidence = (
            state
            if isinstance(state, str)
            else json.dumps(_annotate(state), ensure_ascii=False)
        )
        if image_count:
            marks = evidence.count("<image>")
            if marks == image_count:
                evidence = evidence.replace("<image>", IMAGE_SLOT)
            elif marks:
                raise ValueError(
                    "the state image markers must match the number of images"
                )
            else:
                header = (
                    IMAGE_SLOT
                    if image_count == 1
                    else "\n".join(
                        f"Image {i + 1}: {IMAGE_SLOT}" for i in range(image_count)
                    )
                )
                evidence = header + ("\n\n" + evidence if evidence.strip() else "")
        evidence = evidence if evidence.strip() else "(none)"
        instructions = _text(question.get("instructions")) or (
            "Which answer fits the evidence?" if spec.kind == "noul" else ""
        )
        if not instructions:
            raise ValueError("StartLux choice and score questions require instructions")
        if len(spec.labels) > len(self.token_ids):
            raise ValueError("StartLux question has too many options")
        descriptions = question.get("criteria") or {}
        if spec.kind == "score":
            descriptions = {str(i): value for i, value in enumerate(descriptions)}
        bare = re.compile(
            r"^(?:\(?[A-Za-z][\).]?|\(?\d{1,3}[\).]?|option[_ ]?\d{1,3}|opt[_ ]?\d{1,3})$",
            re.I,
        )
        listed = spec.labels[:-1] if question.get("_extra_none") else spec.labels
        hide = spec.kind == "choice" and all(
            bare.match(label) and _text(descriptions.get(label)) for label in listed
        )
        lines = []
        for i, label in enumerate(spec.labels):
            name = (
                ("yes" if label == "true" else "no") if spec.kind == "noul" else label
            )
            description = _text(descriptions.get(label))
            body = (
                description
                if hide and label in listed
                else name
                if not description or _norm(description) == _norm(name)
                else name + ": " + description
            )
            lines.append(chr(65 + i) + ") " + body)
        content = (
            "Evidence:\n"
            + evidence
            + "\n\nQuestion: "
            + instructions
            + "\nOptions:\n"
            + "\n".join(lines)
        )
        if image_count:
            parts = []
            for index, part in enumerate(content.split(IMAGE_SLOT)):
                if index:
                    parts.append({"type": "image"})
                if part:
                    parts.append({"type": "text", "text": part})
            content = parts
        return [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": content},
        ]

    def render(self, state, spec, question, *, thinking=False, image_count=0):
        messages = self.messages(state, spec, question, image_count=image_count)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        if thinking and not prompt.rstrip().endswith("<think>"):
            prompt += "<think>\n"
        return prompt

    def encode(self, state, spec, question=None, *, image_count=0):
        prompt = self.render(state, spec, question or {}, image_count=image_count)
        return self.tokenizer.encode(
            tokenizable(prompt), add_special_tokens=False
        ), tuple(self.token_ids[: len(spec.labels)])

    def flags(self, spec):
        return protocol.RequestFlag(0)

    def probabilities(self, spec, logits, *, after=False):
        return scoring.softmax(
            [x / (1.0 if after else self.temperatures[spec.kind]) for x in logits]
        )

    def order(self, spec):
        return tuple(range(len(spec.labels)))

    def answer(self, spec, probabilities):
        return KevScorer.answer(self, spec, probabilities)

    def status(self):
        return {"type": self.kind, "generation": True, "gate": 0.6, "mix": 0.75}

    def prepare(
        self,
        app,
        state,
        questions,
        specs,
        controls,
        deadline,
        priority,
        salt,
        media,
        checkpoint,
    ):
        prepared = []
        total_tokens = 0
        for key, spec in specs:
            checkpoint()
            question = questions[key]
            allow_none = question.get("allow_none")
            if allow_none is None:
                allow_none = controls.allow_none
            elif spec.kind == "choice" and type(allow_none) is not bool:
                raise RequestValidationError(
                    [field_error(["questions", key, "allow_none"], "must be a boolean")]
                )
            include_none = spec.kind == "choice" and (allow_none or controls.none_check)
            if spec.kind == "score" and not 2 <= len(spec.labels) <= 10:
                raise RequestValidationError(
                    [
                        field_error(
                            ["questions", key, "criteria"],
                            "StartLux scores need 2 to 10 levels",
                        )
                    ]
                )
            if include_none and any(
                _norm(label) == _norm(NONE) for label in spec.labels
            ):
                raise RequestValidationError(
                    [
                        field_error(
                            ["questions", key, "criteria"],
                            "the None of the above option is added automatically",
                        )
                    ]
                )
            if spec.deterministic and not include_none:
                prepared.append(
                    (key, spec, question, allow_none, include_none, (), False)
                )
                continue
            max_choices = len(self.token_ids) - int(include_none)
            wide = spec.kind == "choice" and len(spec.labels) > max_choices
            groups = _groups(spec.labels) if wide else [spec.labels]
            passes = []
            for group in groups:
                sub, body = _options(spec, question, group, include_none and not wide)
                try:
                    job = self._job(
                        app, state, sub, body, deadline, priority, salt, media
                    )
                except ValueError as error:
                    raise RequestValidationError(
                        [field_error(["questions", key], str(error))]
                    ) from error
                total_tokens += len(job.prompt_tokens)
                if total_tokens > scoring.MAX_SYSTEMONE_TOTAL_TOKENS:
                    raise RequestValidationError(
                        [
                            field_error(
                                ["questions", key],
                                "total prepared question tokens exceed limit",
                            )
                        ]
                    )
                passes.append((sub, body, job))
            prepared.append(
                (key, spec, question, allow_none, include_none, passes, wide)
            )
        return prepared

    def _job(self, app, state, spec, question, deadline, priority, salt, media):
        tokens, letters = self.encode(
            state, spec, question, image_count=len(media[0]) if media else 0
        )
        job = app.score_job(tokens, letters, deadline, priority, salt, media=media)
        if len(job.prompt_tokens) > app.max_context:
            raise ContextLengthError(len(job.prompt_tokens), app.max_context)
        return job

    def respond(self, exchange, body, deadline):
        start = time.perf_counter()
        controls = _controls(body)
        body = _normalize_questions(body)
        prepared = scoring.prepare_systemone(
            exchange.app,
            body,
            deadline=deadline,
            disconnected=exchange.client_disconnected,
            startlux=controls,
        )
        answers = {}
        input_tokens = output_tokens = 0
        first_total = after_total = 0.0
        spans = []
        for key, spec, question, allow_none, include_none, passes, wide in prepared:
            begin = time.perf_counter()
            if not passes:
                answer = scoring.deterministic_answer(spec)
                if controls.details:
                    answer["details"] = _details(
                        {spec.labels[0]: 1.0}, 1.0, 0.0, controls
                    )
                answers[key] = answer
                continue
            media = (
                (passes[0][2].image_owner, passes[0][2].image_pixels)
                if passes[0][2].image_spans
                else None
            )
            first_round = {}
            first_ms = 0.0
            if wide:
                finalists = []
                for group, _, job in passes:
                    values, count, elapsed = self._read(exchange, key, group, job)
                    input_tokens += count
                    first_ms += elapsed
                    first_round.update(zip(group.labels, values, strict=True))
                    finalists.extend(
                        sorted(group.labels, key=first_round.__getitem__, reverse=True)[
                            :3
                        ]
                    )
                limit = len(self.token_ids) - int(include_none)
                if len(finalists) > limit:
                    finalists = sorted(
                        finalists, key=first_round.__getitem__, reverse=True
                    )[:limit]
                current, rendered = _options(spec, question, finalists, include_none)
                first_job = passes[0][2]
                job = self._job(
                    exchange.app,
                    body["state"],
                    current,
                    rendered,
                    deadline,
                    first_job.priority,
                    first_job.cache_salt,
                    media,
                )
            else:
                current, rendered, job = passes[0]
            first, count, elapsed = self._read(exchange, key, current, job)
            input_tokens += count
            first_ms += elapsed
            first_total += first_ms
            details = _details(
                _distribution(current, first, first_round, include_none),
                max(first),
                first_ms,
                controls,
            )
            final = first
            room = exchange.app.max_context - len(job.prompt_tokens) - 64
            if room > 32 and (
                controls.mode == "always"
                or controls.mode == "auto"
                and max(first) < controls.gate
            ):
                thought, span = self._think(
                    exchange,
                    key,
                    body["state"],
                    current,
                    rendered,
                    controls,
                    job,
                    media,
                    room,
                )
                spans.append(span)
                after, after_ms = thought.pop("readout")
                output_tokens += thought["tokens"]
                after_total += after_ms
                final = [
                    (1 - controls.mix) * a + controls.mix * b
                    for a, b in zip(first, after, strict=True)
                ]
                details.update(
                    thought=thought,
                    after_thought={
                        "probabilities": _distribution(
                            current, after, first_round, include_none
                        ),
                        "ms": after_ms,
                    },
                    mix=controls.mix,
                )
            if spec.kind == "choice":
                dist = _distribution(
                    current, final, first_round, include_none, show_none=False
                )
                chosen = max(dist, key=dist.__getitem__)
                answer = {
                    "type": "choice",
                    "choice": chosen,
                    "probabilities": dist,
                    "confidence": 1.0
                    if len(dist) == 1
                    else max(0.0, (len(dist) * dist[chosen] - 1) / (len(dist) - 1)),
                }
                if include_none:
                    answer["none_probability"] = final[-1]
                    if allow_none and final[-1] > max(final[:-1]):
                        answer["choice"] = None
            else:
                answer = self.answer(current, final)
            if controls.details:
                details["total_ms"] = _ms(time.perf_counter() - begin)
                answer["details"] = details
            answers[key] = answer
        response = {
            "model": exchange.app.response_model,
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        }
        if controls.details:
            response["timing_ms"] = {
                "wall": _ms(time.perf_counter() - start),
                "first_pass_sum": first_total,
                "after_thought_sum": after_total,
                "thought_span": _ms(spans[-1][1] - spans[0][0]) if spans else 0.0,
            }
        exchange.send_json(200, response)

    def _read(self, exchange, key, spec, job, *, after=False):
        begin = time.perf_counter()
        exchange.submit(job, key)
        result = scoring.await_done(exchange, job)
        return (
            self.probabilities(spec, result.option_logits, after=after),
            result.prompt_tokens,
            _ms(time.perf_counter() - begin),
        )

    def _think(
        self, exchange, key, state, spec, question, controls, first_job, media, room
    ):
        app = exchange.app
        base = self.render(
            state,
            spec,
            question,
            thinking=True,
            image_count=len(media[0]) if media else 0,
        )
        prompt = base + PREFIX
        seed = int(
            hashlib.sha1(prompt.encode("utf-8", "surrogatepass")).hexdigest()[:8], 16
        )
        budget = min(controls.budget, room)
        text, output, ended, round_no = "", 0, "budget", 0
        begin = time.perf_counter()
        while output < budget:
            tokens = self.tokenizer.encode(
                tokenizable(prompt + text), add_special_tokens=False
            )
            job = app.score_job(
                tokens,
                (),
                first_job.deadline,
                first_job.priority,
                first_job.cache_salt,
                media=media,
            )
            job.max_new_tokens = min(
                1024, budget - output, app.max_context - len(job.prompt_tokens)
            )
            if job.max_new_tokens <= 0:
                break
            job.seed = (seed + 7919 * round_no) % (2**31)
            job.sampling = protocol.SamplingParameters(
                temperature=0.6, top_p=0.95, top_k=20
            )
            job.stop_sequences = ("</think>",)
            exchange.submit(job, key)
            chunks = []
            while (event := exchange.next_event(job))[0] != "done":
                if event[0] == "text":
                    chunks.append(event[1])
            result = event[1]
            text += "".join(chunks)
            output += result.completion_tokens
            if result.reason == "stop":
                ended = "closed"
                break
            if not result.completion_tokens:
                break
            if (cut := loop_cut(text)) is not None:
                text, ended = text[:cut], "loop"
                break
            round_no += 1
        end = time.perf_counter()
        thought = (
            (PREFIX + text).replace("<think>", "", 1).replace("</think>", "").strip()
        )
        seen = "[...]\n" + thought[-4000:] if len(thought) > 4000 else thought
        read_prompt = base + seen + (CUT if ended == "budget" else CLOSE)
        tokens = self.tokenizer.encode(
            tokenizable(read_prompt), add_special_tokens=False
        )
        second = app.score_job(
            tokens,
            first_job.score_tokens,
            first_job.deadline,
            first_job.priority,
            first_job.cache_salt,
            media=media,
        )
        if len(second.prompt_tokens) > app.max_context:
            raise ContextLengthError(len(second.prompt_tokens), app.max_context)
        after, _, elapsed = self._read(exchange, key, spec, second, after=True)
        info = {
            "used": True,
            "tokens": output,
            "ended": ended,
            "generation": "chunked",
            "ms": _ms(end - begin),
            "readout": (after, elapsed),
        }
        if controls.thought_text:
            info["text"] = thought
        return info, (begin, end)


NONE = "None of the above"


def _norm(value):
    return re.sub(r"[\s_\-]+", " ", value).strip().lower()


@dataclass(frozen=True)
class Controls:
    mode: str
    gate: float
    mix: float
    budget: int
    details: bool
    thought_text: bool
    allow_none: bool
    none_check: bool


def _controls(body):
    def get(key, default):
        value = body.get(key)
        return default if value is None else value

    mode = get("think_mode", "auto")
    if isinstance(mode, str):
        mode = {"hybrid": "auto", "plain": "never", "think": "always"}.get(mode, mode)
    gate, mix, budget = get("gate", 0.6), get("mix", 0.75), get("think_budget", 4096)
    errors = []
    if mode not in ("auto", "never", "always"):
        errors.append(field_error(["think_mode"], "must be auto, never, or always"))
    for key, value in (("gate", gate), ("mix", mix)):
        if type(value) not in (int, float) or not 0 <= value <= 1:
            errors.append(field_error([key], "must be a number between zero and one"))
    if type(budget) is not int or budget < 1:
        errors.append(field_error(["think_budget"], "must be a positive integer"))
    flags = {
        key: get(key, default)
        for key, default in (
            ("return_details", get("details", False)),
            ("return_thought", False),
            ("allow_none", False),
            ("none_check", True),
        )
    }
    for key, value in flags.items():
        if type(value) is not bool:
            errors.append(field_error([key], "must be a boolean"))
    if get("thought_generation", "auto") not in ("auto", "chunked"):
        errors.append(
            field_error(
                ["thought_generation"],
                "Splash supports auto or chunked thought generation",
            )
        )
    if errors:
        raise RequestValidationError(errors)
    return Controls(
        mode,
        gate,
        mix,
        min(budget, 4096),
        flags["return_details"] or flags["return_thought"],
        flags["return_thought"],
        flags["allow_none"],
        flags["none_check"],
    )


def _normalize_questions(body):
    questions = body.get("questions")
    if not isinstance(questions, dict):
        return body
    normalized = {}
    for key, question in questions.items():
        if not isinstance(question, dict):
            normalized[key] = question
            continue
        kind = question.get("type", "choice")
        if kind == "bool":
            kind = "noul"
        criteria = question.get("criteria", question.get("options"))
        if kind == "choice" and isinstance(criteria, list):
            criteria = {str(item): None for item in criteria}
        if kind == "score" and isinstance(criteria, dict):
            try:
                keys = sorted(criteria, key=float)
            except ValueError:
                keys = list(criteria)
            criteria = [criteria[key] for key in keys]
        normalized[key] = {
            **question,
            "type": kind,
            "criteria": criteria,
            "instructions": question.get("instructions", question.get("question")),
        }
    return {**body, "questions": normalized}


def _options(spec, question, labels, include_none):
    if spec.kind != "choice":
        return spec, question
    criteria = {label: question["criteria"][label] for label in labels}
    if include_none:
        criteria[NONE] = None
    return replace(spec, labels=tuple(criteria)), {
        **question,
        "criteria": criteria,
        "_extra_none": include_none,
    }


def _groups(labels):
    size, extra = divmod(len(labels), (len(labels) + 24) // 25)
    start = 0
    while start < len(labels):
        end = start + size + bool(extra)
        yield labels[start:end]
        start = end
        extra = max(0, extra - 1)


def _distribution(spec, probabilities, first_round, include_none, *, show_none=True):
    none = probabilities[-1] if include_none else 0.0
    labels = spec.labels[:-1] if include_none else spec.labels
    listed = probabilities[:-1] if include_none else probabilities
    mass = sum(listed)
    dist = {
        key: value / mass if mass else 1 / len(labels)
        for key, value in zip(labels, listed, strict=True)
    }
    if first_round:
        rest = {key: value for key, value in first_round.items() if key not in dist}
        if rest:
            residual = sum(rest.values())
            dist = {key: 0.999 * value for key, value in dist.items()}
            dist.update(
                {
                    key: 0.001 * (value / residual if residual else 1 / len(rest))
                    for key, value in rest.items()
                }
            )
    if include_none and show_none:
        dist = {key: (1 - none) * value for key, value in dist.items()}
        dist[NONE] = none
    return dist


def _ms(seconds):
    return round(seconds * 1000, 3)


def _details(first, top, elapsed, controls):
    return {
        "first_pass": {"probabilities": first, "top": top, "ms": elapsed},
        "thought": {"used": False, "tokens": 0, "ended": "skipped", "ms": 0.0},
        "after_thought": None,
        "gate": controls.gate if controls.mode == "auto" else None,
        "mix": None,
    }


def _periodic(text, low, high, repeats, span):
    for width in range(low, min(high, len(text) // repeats) + 1):
        copies = max(repeats, (span + width - 1) // width)
        if len(text) >= copies * width and text.endswith(text[-width:] * copies):
            return True
    return False


def _looping(text):
    short = re.sub(r"\d+", "#", text[-400:])
    if any(_periodic(s, 1, 80, 4, 60) for s in (short, re.sub(r"\s+", " ", short))):
        return True
    digits = text[-150:]
    if len(digits) == 150 and sum(c.isdigit() for c in digits) >= 135:
        return True
    long = re.sub(r"\s+", " ", re.sub(r"\d+", "#", text[-1600:]))
    if _periodic(long, 81, 400, 3, 0):
        return True
    lines = [
        " ".join(line.split()) for line in text[-1600:].split("\n") if line.strip()
    ]
    words = re.sub(r"\d+", "#", text[-720:]).split()
    return (
        len(lines) > 20
        and len(set(lines[-20:])) <= 8
        or len(words) > 60
        and len(set(words[-60:])) <= 6
    )


def loop_cut(text):
    """StartLux's v3 tail loop detector and cut point, checked per chunk."""
    if not _looping(text):
        return None
    low, high = 0, len(text)
    while high - low > 8:
        mid = (low + high) // 2
        if _looping(text[:mid]):
            high = mid
        else:
            low = mid
    return high
