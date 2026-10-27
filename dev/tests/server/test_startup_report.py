import unittest
from types import SimpleNamespace

from dev.tests.server_fixtures import main_args
from server import startup_report


class StartupReportTests(unittest.TestCase):
    def test_local_control_does_not_imply_model_ready(self):
        readiness = SimpleNamespace(
            max_context_tokens=0, max_concurrent_requests=0, vision=False
        )
        text = "\n".join(
            startup_report.effective_lines(
                {
                    "ready": True,
                    "moe_memory_pressure": {
                        "enabled": True,
                        "mode": "local",
                        "model_ready": False,
                        "growth_allowed": False,
                        "reason": "headroom_observation",
                        "normal_stable_milliseconds": 29999,
                        "block_cooldown_milliseconds": 0,
                    },
                    "memory_actual": {
                        "allocated_bytes": 1234,
                        "process_physical_footprint_bytes": 5678,
                    },
                },
                readiness,
            )
        )
        self.assertIn("MoE memory pressure = local; model_ready=False", text)
        self.assertIn("growth_allowed=False; reason=headroom_observation", text)
        self.assertIn("stable=29999 ms; cooldown=0 ms", text)
        self.assertIn("Metal allocated = 1234 bytes", text)
        self.assertIn("Process physical footprint = 5678 bytes", text)
        self.assertNotIn("Coordination =", text)
        self.assertNotIn("role=", text)

    def test_adaptive_store_report_distinguishes_active_stopped_and_off(self):
        readiness = SimpleNamespace(
            max_context_tokens=262144, max_concurrent_requests=1, vision=False
        )
        for enabled, state in ((True, "adaptive"), (True, "stopped"), (False, "off")):
            status = {
                "ready": True,
                "expert_ple_adaptive": {
                    "enabled": enabled,
                    "state": state,
                    "ratio": "1:20",
                    "ple_budget_bytes": 536870912,
                    "ple_maximum_bytes": 1022611260,
                    "failure": "allocation refused" if state == "stopped" else "",
                },
            }
            text = "\n".join(startup_report.effective_lines(status, readiness))
            self.assertIn(f"Expert/PLE adaptive = {state}", text)
            self.assertIn(
                "ratio=1:20; PLE=536870912 bytes; maximum=1022611260 bytes", text
            )
            if state == "stopped":
                self.assertIn("allocation refused", text)

    def test_expert_report_uses_weight_bytes_and_preserves_unknown(self):
        readiness = SimpleNamespace(
            max_context_tokens=8192, max_concurrent_requests=1, vision=False
        )
        report = {
            "layout": "addressed",
            "weight_bytes": 1000,
            "resident_layout_bytes": 1100,
            "contiguous_floor_bytes": 600,
            "resizable_floor_bytes": 700,
            "resizable_full_bytes": 1200,
            "budget_bytes": 900,
            "allocated_bytes": 850,
            "auxiliary_bytes": 350,
            "capacity_weight_bytes": 500,
            "resident_weight_bytes": 250,
            "capacity_ratio": 0.5,
            "resident_ratio": 0.25,
            "minimum_required_bytes": None,
            "current_floor_bytes": 700,
        }
        text = "\n".join(
            startup_report.effective_lines(
                {
                    "ready": True,
                    "expert_storage": report,
                    "memory_governor": {
                        "charged_bytes": 1234,
                        "untracked_reserve_bytes": 30,
                        "local_reserved_bytes": 20,
                    },
                },
                readiness,
            )
        )
        for expected in (
            "Expert storage = addressed; full_weights=1000 bytes",
            "contiguous_floor=600 bytes; resizable_floor=700 bytes",
            "budget=900 bytes; allocated=850 bytes; auxiliary=350 bytes",
            "capacity=500 bytes (50.00%); resident=250 bytes (25.00%)",
            "current_floor=700 bytes; minimum_required=unknown",
            "Local memory charge = 1234 bytes; untracked reserve=30 bytes; pending local admission=20 bytes",
        ):
            self.assertIn(expected, text)
        for status in ({}, {"expert_storage": None}):
            self.assertIn(
                "Expert storage = unknown",
                "\n".join(startup_report.effective_lines(status, readiness)),
            )

    def test_complete_configuration_masks_secrets_and_does_not_enable_compact(self):
        args = main_args(
            api_key="private-secret", compact_draft_ranking="/missing/rank"
        )
        text = "\n".join(startup_report.configuration_lines(args))
        self.assertIn("--api-key = on (redacted)", text)
        self.assertNotIn("private-secret", text)
        self.assertIn("--compact-draft-head = off", text)
        self.assertIn("--compact-draft-ranking = /missing/rank", text)
        self.assertIn("--ane-ffn-share = auto", text)
        self.assertIn("--binary = splash", text)
        self.assertIn("--max-memory = auto", text)
        self.assertIn("--expert-ple-memory = auto", text)
        self.assertIn("--expert-memory = auto", text)
        self.assertIn("--ple-memory = auto", text)
        self.assertIn("--sparse-kv-memory = auto", text)
        self.assertIn("--decode-share = default (0.5)", text)
        self.assertIn("--request-timeout = off", text)
        self.assertIn("--allow-idle-sleep = off", text)
        # New shared options must also appear without a separate reporting list.
        from server.serve_options import SERVE_OPTIONS

        for option in SERVE_OPTIONS:
            self.assertEqual(text.count(option.flag + " = "), 1)

    def test_actual_status_distinguishes_backoff_invalid_reference_and_validation(self):
        status = {
            "ready": True,
            "ane_ffn": {
                "state": "stopped",
                "share": 0.4118,
                "minimum_rows": 512,
                "reason": "evaluation timed out",
                "recovery": {
                    "enabled": True,
                    "phase": "unavailable",
                    "reason": "no reference",
                },
                "power_reference": {
                    "valid": False,
                    "reason": "background power too high",
                },
            },
            "startup_features": {
                "drafter": "dflash2",
                "draft_layers": 5,
                "draft_block_size": 8,
                "draft_tokens": 7,
                "prefill_token_budget": 2048,
                "optional": {
                    "prompt_lookup": True,
                    "lookup_reason": "",
                    "compact_draft_head": True,
                    "compact_validated": True,
                    "compact_capacity": 98304,
                    "compact_validation_maximum_error": 0,
                    "compact_coverage_fallbacks": 2,
                    "compact_request_fallbacks": 1,
                    "fast_topk": False,
                },
            },
        }
        readiness = SimpleNamespace(
            max_context_tokens=262144, max_concurrent_requests=4, vision=False
        )
        text = "\n".join(startup_report.effective_lines(status, readiness))
        self.assertIn("ANE = fallback", text)
        self.assertIn("evaluation timed out", text)
        self.assertIn("ANE power reference = unavailable", text)
        self.assertIn("background power too high", text)
        self.assertIn("ANE recovery = unavailable", text)
        self.assertIn("Compact = validated", text)
        self.assertIn("coverage_fallbacks=2", text)
        self.assertIn("request_fallbacks=1", text)
        self.assertIn("Drafter = on; kind=dflash2; layers=5; block=8; tokens=7", text)
        self.assertIn("Fast top-k = off", text)

    def test_missing_or_stale_status_is_not_reported_as_enabled(self):
        readiness = SimpleNamespace(
            max_context_tokens=8192, max_concurrent_requests=1, vision=True
        )
        for status in (None, {}, {"ready": True, "transport": {"status_stale": True}}):
            with self.subTest(status=status):
                text = "\n".join(startup_report.effective_lines(status, readiness))
                self.assertIn("ANE = unknown", text)
                self.assertIn("Compact = unknown", text)
                self.assertIn("Drafter = unknown", text)
                self.assertIn("Experimental quantized FFN = unknown", text)
                self.assertIn("Context = 8192", text)
                self.assertIn("Vision = on", text)

    def test_quantized_ffn_reports_loaded_state_not_requested_flag(self):
        readiness = SimpleNamespace(
            max_context_tokens=8192, max_concurrent_requests=1, vision=False
        )
        args = main_args(experimental_quantized_ffn="on")
        for enabled, cache_bytes in ((True, 0), (False, 4529848320)):
            status = {
                "ready": True,
                "startup_features": {
                    "experimental_quantized_ffn": enabled,
                    "ffn_cache_bytes": cache_bytes,
                },
            }
            text = "\n".join(startup_report.effective_lines(status, readiness, args))
            self.assertIn(
                f"Experimental quantized FFN = {'on' if enabled else 'off'}", text
            )
            self.assertIn(f"FFN FP16 cache = {cache_bytes} bytes", text)

    def test_log_values_cannot_inject_additional_lines(self):
        text = "\n".join(
            startup_report.configuration_lines(
                main_args(served_model_name=["line\nforged\r"])
            )
        )
        self.assertIn("line\\nforged\\r", text)
        self.assertNotIn("line\nforged", text)

    def test_startup_error_identifies_stage_and_redacts_key(self):
        line = startup_report.failure_line(
            "compact validation",
            ValueError("bad private-key\nrank"),
            main_args(api_key="private-key"),
        )
        self.assertIn("Startup state = error; stage=compact validation", line)
        self.assertIn("later stages unconfirmed", line)
        self.assertNotIn("private-key", line)
        self.assertNotIn("\n", line)

    def test_unvalidated_compact_is_an_error_and_lookup_requires_drafter(self):
        status = {
            "ready": True,
            "startup_features": {
                "drafter": "none",
                "optional": {
                    "prompt_lookup": True,
                    "compact_draft_head": True,
                    "compact_validated": False,
                },
            },
        }
        text = "\n".join(
            startup_report.effective_lines(
                status,
                SimpleNamespace(
                    max_context_tokens=8192, max_concurrent_requests=1, vision=False
                ),
            )
        )
        self.assertIn("Compact = error (not validated)", text)
        self.assertIn("Prompt lookup = unavailable (no drafter)", text)

    def test_absolute_power_recovery_does_not_require_valid_relative_reference(self):
        status = {
            "ready": True,
            "ane_ffn": {
                "state": "split",
                "power_reference": {"valid": False},
                "recovery": {"enabled": True, "phase": "running"},
            },
        }
        ready = SimpleNamespace(
            max_context_tokens=8192, max_concurrent_requests=1, vision=False
        )
        absolute = "\n".join(startup_report.effective_lines(status, ready, main_args()))
        relative = "\n".join(
            startup_report.effective_lines(
                status, ready, main_args(ane_recovery_max_power_percent=20)
            )
        )
        self.assertIn("ANE recovery = on", absolute)
        self.assertIn("ANE recovery = unavailable", relative)
