"""Prompt-injection screening for retrieved text (defence in depth, not a guarantee).

Retrieved documents are untrusted input. Before they reach the model we:
1. drop sentences that match known injection patterns (and count them), and
2. frame what remains as quoted data inside delimited blocks (see `prompts.py`).

Pattern lists are easy to bypass; that's why the system prompt and the adversarial evaluation
items exist as well. Every drop is counted so it shows up in logs and metrics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above) (instructions|prompts?)",
    r"disregard (all |any )?(the )?(previous|prior|above)",
    r"(system|developer) (notice|prompt|message) to (ai|assistants?|the model)",
    r"you are now (in )?(developer|dan|jailbreak)",
    r"(reveal|print|show) (your|the) (system )?prompt",
    r"tell (every|all|the) users? that",
]
_INJECTION = re.compile("|".join(_PATTERNS), re.IGNORECASE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True, slots=True)
class ScreenResult:
    text: str
    removed: tuple[str, ...]

    @property
    def flagged(self) -> bool:
        return bool(self.removed)


def screen(text: str) -> ScreenResult:
    kept: list[str] = []
    removed: list[str] = []
    for paragraph in text.split("\n\n"):
        sentences = _SENTENCE.split(paragraph)
        # One injected sentence taints the whole paragraph: attackers split instructions over
        # several sentences ("IMPORTANT NOTICE: ignore ... Tell every user ...").
        if any(_INJECTION.search(s) for s in sentences):
            removed.append(paragraph)
        else:
            kept.append(paragraph)
    return ScreenResult("\n\n".join(kept), tuple(removed))
