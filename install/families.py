"""The model families Splash serves and their default draft repositories.

A target's family is the one the engine finds its configuration to describe,
by the rules it holds every start to (upstream.check_model), never its
repository's name. Legacy Splash packages hold these same families.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelFamily:
    # The family's name, as the engine names it (each target layout's family,
    # runtime/model/Qwen3_8.hpp and Qwen3_6Moe.hpp).
    name: str
    # The repository of the DFlash2 checkpoint trained for the family, as its
    # release publishes it: config.json and BF16 safetensors. Installations
    # follow its default branch as they follow the target's.
    draft_repo: str | None


FAMILIES = (
    ModelFamily("Qwen3.5-4B", None),
    ModelFamily("Qwen3.8-27B", "incoai/Qwen3.8-27B-DFlash2"),
    ModelFamily("Qwen3.6-35B-A3B", "incoai/Qwen3.6-35B-A3B-DFlash2"),
)


# Qwen3.5-4B remains ordinary decoding by default. Explicit selection of its
# standard DFlash release uses the audited checkpoint revision.
QWEN35_4B_DFLASH_REPO = "z-lab/Qwen3.5-4B-DFlash"
QWEN35_4B_DFLASH_REVISION = "9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf"


def named(name):
    """The family called name, or None."""
    return next((family for family in FAMILIES if family.name == name), None)
