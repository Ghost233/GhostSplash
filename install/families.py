"""The model families Splash serves and their default draft repositories.

A target's family is the one the engine finds its configuration to describe,
by the rules it holds every start to (upstream.check_model), never its
repository's name.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelFamily:
    # The family's name, as the engine names it (each target layout's family,
    # runtime/model/Qwen3_8.hpp, Qwen3_6Moe.hpp and Qwen4Exp.hpp).
    name: str
    # The repository of the DFlash2 checkpoint trained for the family, as its
    # release publishes it: config.json and BF16 safetensors. Installations
    # follow its default branch as they follow the target's. None says no
    # default is released; only NO_DRAFT_FAMILIES may serve without one.
    draft_repo: str | None


FAMILIES = (
    ModelFamily("Qwen3.5-4B", None),
    ModelFamily("Kev-4B", None),
    ModelFamily("Kev-0.8B", None),
    ModelFamily("D3-lite", None),
    ModelFamily("D3-mini", None),
    ModelFamily("Julia-1", None),
    ModelFamily("Laya", None),
    ModelFamily("Qwen3.8-27B", "incoai/Qwen3.8-27B-DFlash2"),
    ModelFamily("Qwen3.6-35B-A3B", "incoai/Qwen3.6-35B-A3B-DFlash2"),
    ModelFamily("Qwen3.8-Flash-Next", "incoai/Qwen3.8-Flash-Next-DFlash2"),
)

NO_DRAFT_FAMILIES = frozenset(
    {"Qwen3.5-4B", "Kev-4B", "Kev-0.8B", "D3-lite", "D3-mini", "Julia-1", "Laya"}
)


# Qwen3.5-4B remains ordinary decoding by default. Explicit selection of its
# standard DFlash release uses the audited checkpoint revision.
QWEN35_4B_DFLASH_REPO = "z-lab/Qwen3.5-4B-DFlash"
QWEN35_4B_DFLASH_REVISION = "9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf"


def named(name):
    """The family called name, or None."""
    return next((family for family in FAMILIES if family.name == name), None)
