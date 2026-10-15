#!/usr/bin/env python3
"""Verify exact 32768-token cold input and summary facts at the start, middle
and end. Run only against the dedicated, already-started 4B server. Writes
its full request, response and native timings even when a business check fails.

    .venv/bin/python -m dev.tests.summary_32k_real --output build/summary-32k.json
"""

import argparse
import hashlib
import json
import re
import time
import unicodedata
import uuid
from pathlib import Path

from transformers import AutoTokenizer

from dev.tests import smoke_real as smoke
from install import models

ROOT = Path(__file__).resolve().parents[2]
MODEL = "unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL"
INPUT_TOKENS = 32768
FACTS = {
    "start": ["北辰平台", "480万元", "2027年3月18日", "林悦"],
    "middle": ["远帆迁移", "73套", "2027年6月9日", "周衡"],
    "end": ["星桥审计", "19项", "2027年9月26日", "沈宁"],
}


def missing_summary_facts(summary):
    normalized = "".join(unicodedata.normalize("NFKC", summary).split())
    missing = {
        position: [
            fact
            for fact in facts
            if not re.search(
                (r"(?<!\d)" if fact[0].isdigit() else "") + re.escape(fact), normalized
            )
        ]
        for position, facts in FACTS.items()
    }
    return {position: facts for position, facts in missing.items() if facts}


def content_at_size(tokenizer, nonce):
    notes = "\n".join(
        f"第{i}次例会：团队复核需求、接口契约和数据字段，保留变更原因与验收记录。"
        "测试环境与生产环境分别核对，迁移前检查备份，变更后观察错误率和响应时间。"
        "风险清单区分已关闭、处理中与待确认事项；例会不修改专项预算和批准日期。"
        for i in range(1, 65)
    )
    filler = tokenizer.encode(notes, add_special_tokens=False)
    rows = INPUT_TOKENS - 400
    for _ in range(16):
        first, second = rows // 2, rows - rows // 2
        repeated = filler * (max(first, second) // len(filler) + 1)
        content = (
            f"档案标识：{nonce}\n以下是年度项目档案，请阅读全部内容后做总结。\n"
            "开篇专项：北辰平台批准预算480万元，正式上线日期2027年3月18日，负责人林悦。\n"
            + tokenizer.decode(repeated[:first])
            + "\n中段专项：远帆迁移覆盖73套系统，切换日期2027年6月9日，负责人周衡。\n"
            + tokenizer.decode(repeated[:second])
            + "\n末尾专项：星桥审计发现19项待整改问题，整改截止2027年9月26日，负责人沈宁。\n"
            "请用中文概括档案的工作方法，并分别总结北辰平台、远帆迁移和星桥审计。"
            "每个专项必须保留原文数字、日期和负责人；不要编造档案中没有的结果。"
        )
        tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
            reasoning_effort="none",
        )
        difference = len(tokens) - INPUT_TOKENS
        if difference == 0:
            return content
        rows -= difference
    raise ValueError("could not construct an exact 32768-token summary input")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--request-from",
        type=Path,
        help="reuse an earlier report's identical input for the baseline comparison",
    )
    parser.add_argument(
        "--kv-format",
        choices=("int8", "bf16", "q4h", "q4_0", "q4_0_unrotated"),
        default="int8",
    )
    args = parser.parse_args()
    if args.request_from:
        earlier = json.loads(args.request_from.read_text())
        body = earlier["request"]
        smoke.require(
            earlier["expected_facts"] == FACTS and body["model"] == MODEL,
            "comparison input is not this 4B summary fixture",
        )
    else:
        selection = models.Selection.of(
            ROOT / "install/models-local", MODEL, language_only=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            selection.link / "tokenizer", local_files_only=True
        )
        content = content_at_size(tokenizer, uuid.uuid4().hex)
        body = smoke.chat_body(MODEL, content, max_completion_tokens=1024)
    result = {
        "input_tokens": INPUT_TOKENS,
        "requested_kv_format": args.kv_format,
        "request_sha256": hashlib.sha256(
            json.dumps(body, ensure_ascii=False).encode()
        ).hexdigest(),
        "request": body,
        "expected_facts": FACTS,
        "business_checks_passed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    code, applied = smoke.request(8011, "POST", "/apply-template", body)
    smoke.require(code == 200, f"template rendering failed: {applied}")
    code, encoded = smoke.request(
        8011, "POST", "/tokenize", {"content": applied["prompt"]}
    )
    smoke.require(
        code == 200 and len(encoded["tokens"]) == INPUT_TOKENS,
        "server tokenizer did not confirm exactly 32768 input tokens",
    )
    status_code, before = smoke.request(8011, "GET", "/status")
    smoke.require(status_code == 200, f"cannot read pre-request status: {before}")
    result["status_before"] = before
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    smoke.validate_status(before, args.kv_format)
    started = time.monotonic()
    try:
        code, response = smoke.request(
            8011, "POST", "/v1/chat/completions", body, timeout=1800
        )
        elapsed = time.monotonic() - started
        result.update(http_status=code, http_wall_seconds=elapsed, response=response)
        smoke.require(code == 200, f"summary request failed: {response}")
        status_code, after = smoke.request(8011, "GET", "/status")
        result["status_after"] = after
        smoke.require(status_code == 200, f"cannot read post-request status: {after}")
        smoke.require(
            response["usage"]["prompt_tokens"] == INPUT_TOKENS, "input was truncated"
        )
        smoke.require(
            response["metrics"]["cache"]["matched_tokens"] == 0,
            "cold request reused a prefix",
        )
        smoke.require(
            response["metrics"]["prefill"]["tokens"] == INPUT_TOKENS,
            "prefill omitted input rows",
        )
        summary = response["choices"][0]["message"]["content"]
        missing = missing_summary_facts(summary)
        result["summary"] = summary
        result["prefill_metrics"] = response["metrics"]["prefill"]
        result["memory_actual_before"] = before.get("memory_actual")
        result["memory_actual_after"] = after.get("memory_actual")
        result["missing_facts"] = missing
        smoke.require(not missing, f"summary lost document facts: {missing}")
        smoke.require(
            after["state"]["active_lanes"] == 0 and after["kv"]["pages_active"] == 0,
            "summary request left active model resources",
        )
        result["business_checks_passed"] = True
    except Exception as error:
        result["failure"] = str(error)
        raise
    finally:
        result.setdefault("http_wall_seconds", time.monotonic() - started)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "input_tokens": INPUT_TOKENS,
                "http_wall_seconds": elapsed,
                "metrics": response["metrics"],
                "facts": "PASS",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
