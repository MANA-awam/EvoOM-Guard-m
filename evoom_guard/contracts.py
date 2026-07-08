# ─────────────────────────────────────────────────────────────────────────────
# Copyright (c) 2026 Mana Alharbi (مانع الحربي). All rights reserved.
# Source-available — see LICENSE for permitted use.
# Sole owner & author: Mana Alharbi (مانع الحربي).
# ─────────────────────────────────────────────────────────────────────────────
"""Shared contracts (extracted from EvoOM's ``evoom_contracts``).

The single interface the judge rests on: a verifier *executes and measures*;
it never trusts a model's opinion of its own output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict, runtime_checkable


class Problem(TypedDict, total=False):
    """A problem definition handed to a verifier.

    Fields are domain-flavoured; only ``name`` is treated as universal. Extra
    keys are allowed because ``total=False``.
    """

    name: str
    signature: str
    description: str
    tests: list[str]
    depends_on: list[str]


@dataclass
class VerdictResult:
    """The structured result returned by every verifier.

    The whole design rests on this object being produced by *objective*
    measurement, never by the model's opinion of its own output.
    """

    passed: bool
    """Did the hypothesis fully pass?"""

    score: float
    """Numeric score used for ranking, in [0..1]."""

    diagnostics: str
    """Diagnostic trace the caller (or a generator) learns from."""

    artifact: dict[str, Any] = field(default_factory=dict)
    """Extra outputs (logs, metrics)."""


@runtime_checkable
class Verifier(Protocol):
    """The unified verifier interface.

    Golden rule: a verifier must never trust the model. It executes, measures,
    and returns facts.
    """

    domain: str

    def verify(self, hypothesis: str, problem: Problem) -> VerdictResult:
        """Test ``hypothesis`` against ``problem`` and return an objective verdict."""
        ...
