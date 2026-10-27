"""Startup options and confirmed native state, without loading model files."""

from .serve_options import SERVE_OPTIONS


def _text(value):
    return str(value).replace("\r", "\\r").replace("\n", "\\n")


def configuration_lines(args):
    yield "Startup configuration · on/off are configured flags; effective state follows Ready"
    for flag, value in (
        ("MODEL_DIRECTORY", args.model_root),
        ("--model", args.model),
        ("--chat-template", args.chat_template),
        ("--binary", args.binary),
        ("--port", args.port),
    ):
        yield f"  {flag} = {_text(value)}"
    automatic = {
        "max_memory",
        "expert_ple_memory",
        "expert_memory",
        "host_memory_reserve",
        "ple_memory",
        "sparse_kv_memory",
        "max_context",
        "ane_ffn_share",
        "ane_ffn_minimum_rows",
        "default_reasoning_effort",
    }
    for option in SERVE_OPTIONS:
        value = getattr(args, option.dest)
        if option.secret:
            shown = "on (redacted)" if value else "off"
        elif isinstance(value, bool):
            shown = "on" if value else "off"
        elif value is None:
            shown = "auto" if option.dest in automatic else "off / unset"
            if option.dest == "idle_release":
                shown = "auto (default 600 s)"
            elif option.dest == "decode_share":
                shown = "default (0.5)"
        elif isinstance(value, list):
            shown = ", ".join(_text(option.text(item)) for item in value) or "none"
        else:
            shown = _text(option.text(value))
        yield f"  {option.flag} = {shown}"


def effective_lines(status, readiness, args=None):
    yield "Startup effective state · native snapshot; control mode and model readiness are separate"
    yield f"  Context = {readiness.max_context_tokens} tokens"
    yield f"  Native concurrent capacity = {readiness.max_concurrent_requests} (queue limit is separate)"
    yield f"  Vision = {'on' if readiness.vision else 'off'}"
    status = status if isinstance(status, dict) else {}
    transport = status.get("transport", {})
    stale = isinstance(transport, dict) and transport.get("status_stale") is True
    if stale:
        yield f"  Status = unknown (stale snapshot); {_text(transport.get('error', 'refresh pending'))}"
        status = {}
    elif not status:
        yield "  Status = unknown (native snapshot unavailable)"
    elif status.get("ready") is not True:
        yield "  Status = not ready; inspect /status and the engine error log"

    ane = status.get("ane_ffn", {})
    ane_state = {"split": "on", "off": "off", "stopped": "fallback"}.get(
        ane.get("state"), "unknown"
    )
    yield (
        f"  ANE = {ane_state}; share={ane.get('share', 'unknown')}; "
        f"minimum_rows={ane.get('minimum_rows', 'unknown')}; reason={_text(ane.get('reason') or 'none')}"
    )
    reference = ane.get("power_reference", {})
    valid = reference.get("valid")
    reference_state = (
        "valid"
        if valid is True
        else "unavailable"
        if valid is False
        else "off / unreported"
    )
    yield (
        f"  ANE power reference = {reference_state}; watts={reference.get('reference_watts', 'unknown')}; "
        f"reason={_text(reference.get('reason') or 'none')}"
    )
    recovery = ane.get("recovery", {})
    phase = recovery.get("phase", "unknown")
    recovery_state = "off / unreported"
    if recovery.get("enabled") is True:
        needs_reference = (
            args is None or args.ane_recovery_max_power_percent is not None
        )
        unavailable = phase in (
            "unavailable",
            "reference_unavailable",
            "fault_stopped",
            "unknown_power",
        )
        recovery_state = (
            "unavailable"
            if unavailable or (needs_reference and valid is False)
            else "on"
        )
    yield (
        f"  ANE recovery = {recovery_state}; phase={_text(phase)}; "
        f"reason={_text(recovery.get('reason') or 'none')}"
    )

    pressure = status.get("moe_memory_pressure") or {}
    yield (
        f"  MoE memory pressure = {_text(pressure.get('mode', 'unknown'))}; "
        f"model_ready={pressure.get('model_ready', 'unknown')}; "
        f"growth_allowed={pressure.get('growth_allowed', 'unknown')}; "
        f"reason={_text(pressure.get('reason') or 'none')}"
    )
    if pressure.get("enabled") is True:
        yield (
            f"  MoE recovery = stable={pressure.get('normal_stable_milliseconds', 'unknown')} ms; "
            f"cooldown={pressure.get('block_cooldown_milliseconds', 'unknown')} ms"
        )

    adaptive = status.get("expert_ple_adaptive") or {}
    enabled = adaptive.get("enabled")
    adaptive_state = (
        adaptive.get("state", "on")
        if enabled is True
        else "off"
        if enabled is False
        else "unknown"
    )
    yield (
        f"  Expert/PLE adaptive = {adaptive_state}; ratio={_text(adaptive.get('ratio', 'unknown'))}; "
        f"PLE={adaptive.get('ple_budget_bytes', 'unknown')} bytes; "
        f"maximum={adaptive.get('ple_maximum_bytes', 'not applied' if enabled is False else 'unknown')} bytes; "
        f"reason={_text(adaptive.get('failure') or adaptive.get('reason') or 'none')}"
    )

    experts = status.get("expert_storage") or {}

    def expert_value(key):
        value = experts.get(key)
        return "unknown" if value is None else value

    def expert_ratio(key):
        value = experts.get(key)
        return "unknown" if value is None else f"{100 * value:.2f}%"

    yield (
        f"  Expert storage = {expert_value('layout')}; "
        f"full_weights={expert_value('weight_bytes')} bytes; "
        f"resident_layout={expert_value('resident_layout_bytes')} bytes; "
        f"contiguous_floor={expert_value('contiguous_floor_bytes')} bytes; "
        f"resizable_floor={expert_value('resizable_floor_bytes')} bytes; "
        f"resizable_full={expert_value('resizable_full_bytes')} bytes"
    )
    yield (
        f"  Expert allocation = budget={expert_value('budget_bytes')} bytes; "
        f"allocated={expert_value('allocated_bytes')} bytes; "
        f"auxiliary={expert_value('auxiliary_bytes')} bytes; "
        f"current_floor={expert_value('current_floor_bytes')} bytes; "
        f"minimum_required={expert_value('minimum_required_bytes')}"
    )
    yield (
        f"  Expert weights = capacity={expert_value('capacity_weight_bytes')} bytes "
        f"({expert_ratio('capacity_ratio')}); "
        f"resident={expert_value('resident_weight_bytes')} bytes "
        f"({expert_ratio('resident_ratio')}); hit rate is reported per use in /status"
    )

    identity = status.get("identity", {})
    yield f"  KV = {_text(identity.get('kv', {}).get('format', 'unknown'))}"
    yield f"  Build = {_text(identity.get('cache', {}).get('build_id', 'unknown'))}"
    governor = status.get("memory_governor", {})
    yield f"  Metal budget = {governor.get('limit_bytes', 'unknown')} bytes"
    actual = status.get("memory_actual", {})
    yield f"  Metal allocated = {actual.get('allocated_bytes', 'unknown')} bytes"
    footprint = actual.get("process_physical_footprint_bytes")
    yield f"  Process physical footprint = {'unknown' if footprint is None else footprint} bytes"
    yield (
        f"  Local memory charge = {governor.get('charged_bytes', 'unknown')} bytes; "
        f"untracked reserve={governor.get('untracked_reserve_bytes', 'unknown')} bytes; "
        f"pending local admission={governor.get('local_reserved_bytes', 'unknown')} bytes"
    )
    if status.get("model_kind") == "modern-bert":
        encoder = status.get("features", {}).get("encoder_option_queries")
        yield f"  Encoder option queries = {'on' if encoder is True else 'off' if encoder is False else 'unknown'}"
        yield "  Encoder state = full bidirectional sequence; causal prefix cache off"
    features = status.get("startup_features", {})
    drafter = features.get("drafter", "unknown")
    draft_state = (
        "off" if drafter == "none" else "unknown" if drafter == "unknown" else "on"
    )
    yield (
        f"  Drafter = {draft_state}; kind={_text(drafter)}; "
        f"layers={features.get('draft_layers', 'unknown')}; "
        f"block={features.get('draft_block_size', 'unknown')}; tokens={features.get('draft_tokens', 'unknown')}"
    )
    yield f"  Prefill block = {features.get('prefill_token_budget', 'unknown')} tokens (compiled)"
    yield f"  Prefill warmup geometry = {_text(status.get('warmup', {}).get('detail', 'unknown'))}"
    quantized_ffn = features.get("experimental_quantized_ffn")
    yield (
        "  Experimental quantized FFN = "
        + (
            "on"
            if quantized_ffn is True
            else "off"
            if quantized_ffn is False
            else "unknown"
        )
    )
    yield f"  FFN FP16 cache = {features.get('ffn_cache_bytes', 'unknown')} bytes"
    optional = features.get("optional", {})
    for label, key, reason in (
        ("Prompt lookup", "prompt_lookup", "lookup_reason"),
        ("Fast top-k", "fast_topk", "fast_topk_reason"),
    ):
        enabled = optional.get(key)
        state = (
            "on (conditional)"
            if enabled is True
            else "off"
            if enabled is False
            else "unknown"
        )
        if key == "prompt_lookup" and enabled is True and drafter == "none":
            state = "unavailable (no drafter)"
        yield f"  {label} = {state}; reason={_text(optional.get(reason) or 'none')}"
    compact = optional.get("compact_draft_head")
    state = "off" if compact is False else "unknown"
    if compact is True:
        state = (
            "validated"
            if optional.get("compact_validated") is True
            else "error (not validated)"
        )
    yield (
        f"  Compact = {state}; capacity={optional.get('compact_capacity', 'unknown')}; "
        f"maximum_error={optional.get('compact_validation_maximum_error', 'unknown')}; "
        f"coverage_fallbacks={optional.get('compact_coverage_fallbacks', 'unknown')}; "
        f"request_fallbacks={optional.get('compact_request_fallbacks', 'unknown')}; "
        f"reason={_text(optional.get('compact_reason') or 'none')}"
    )


def failure_line(stage, error, args):
    message = _text(error)
    # A diagnostic must not reveal a supplied API key, including an exception
    # whose message embeds one. Startup errors carry no request content.
    if args.api_key:
        message = message.replace(_text(args.api_key), "<redacted>")
    return f"Error · {message} · Startup state = error; stage={stage}; later stages unconfirmed"
