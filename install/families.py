"""The model families Splash serves, their default drafts and chat formats.

A target's family is the one the engine finds its configuration to describe,
by the rules it holds every start to (upstream.check_model), never its
repository's name.
"""

from __future__ import annotations

from dataclasses import dataclass

from server.chat_format import FLINT, QWEN, ChatFormat


@dataclass(frozen=True)
class ModelFamily:
    # The family's name, as the engine names it (each target layout's family,
    # runtime/model/Qwen3_8.hpp, Qwen3_6Moe.hpp, Qwen4Exp.hpp and Flint.hpp).
    name: str
    # The repository of the DFlash2 checkpoint trained for the family, as its
    # release publishes it: config.json and BF16 safetensors. Installations
    # follow its default branch as they follow the target's. None says no
    # default is released; only NO_DRAFT_FAMILIES may serve without one.
    draft_repo: str | None
    # How the family's model reasons and calls tools, and the tokens it
    # shares with the engine (server/chat_format.py).
    chat_format: ChatFormat
    # Whether the family's models have a vision tower, which an installation
    # loads from the target repository unless --language-only. The models of
    # a family without one serve text only, whatever their repository holds.
    vision: bool = True


FAMILIES = (
    ModelFamily("Qwen3.5-4B", None, QWEN),
    ModelFamily("Kev-4B", None, QWEN),
    ModelFamily("Kev-0.8B", None, QWEN),
    ModelFamily("D3-lite", None, QWEN),
    ModelFamily("D3-mini", None, QWEN),
    ModelFamily("Julia-1", None, QWEN),
    ModelFamily("Laya", None, QWEN),
    ModelFamily("Qwen3.8-27B", "incoai/Qwen3.8-27B-DFlash2", QWEN),
    ModelFamily("Qwen3.6-35B-A3B", "incoai/Qwen3.6-35B-A3B-DFlash2", QWEN),
    # Its fine-tune Ornith-1.5-122B is the family too, and the draft
    # released for it the family's.
    ModelFamily("Qwen3.5-122B-A10B", "incoai/Ornith-1.5-122B-DFlash2", QWEN),
    ModelFamily("Qwen3.8-Flash-Next", "incoai/Qwen3.8-Flash-Next-DFlash2", QWEN),
    # It has no vision tower.
    ModelFamily("Flint", "incoai/Flint-DFlash2", FLINT, vision=False),
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
