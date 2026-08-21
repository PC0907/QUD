"""Label taxonomy for the CLARITY task (Thomas et al., 2024).

Two-level hierarchy:
  Level 1 (clarity):  Clear Reply | Ambivalent | Clear Non-Reply
  Level 2 (evasion):  9 fine-grained categories, each nested under exactly
                      one clarity label.

This module is the single source of truth for label names, normalization,
the evasion->clarity mapping, and the QUD-relation signature of each
evasion category used by the rule-based classifier.
"""

from __future__ import annotations

from enum import Enum


# ---------------------------------------------------------------------------
# Canonical label names (as they appear in ailsntua/QEvasion)
# ---------------------------------------------------------------------------

CLARITY_LABELS = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]

EVASION_LABELS = [
    "Explicit",
    "Implicit",
    "Dodging",
    "General",
    "Deflection",
    "Partial/half-answer",
    "Declining to answer",
    "Claims ignorance",
    "Clarification",
]

CLARITY2ID = {l: i for i, l in enumerate(CLARITY_LABELS)}
EVASION2ID = {l: i for i, l in enumerate(EVASION_LABELS)}
ID2CLARITY = {i: l for l, i in CLARITY2ID.items()}
ID2EVASION = {i: l for l, i in EVASION2ID.items()}

# Evasion -> clarity mapping (the taxonomy hierarchy). Deriving clarity
# from a predicted evasion label through this dict is "upward mapping",
# the strategy that won SemEval-2026 Task 6 Subtask 1.
EVASION_TO_CLARITY = {
    "Explicit": "Clear Reply",
    "Implicit": "Ambivalent",
    "Dodging": "Ambivalent",
    "General": "Ambivalent",
    "Deflection": "Ambivalent",
    "Partial/half-answer": "Ambivalent",
    "Declining to answer": "Clear Non-Reply",
    "Claims ignorance": "Clear Non-Reply",
    "Clarification": "Clear Non-Reply",
}

# Aliases observed across dataset versions / papers.
_CLARITY_ALIASES = {
    "ambivalent reply": "Ambivalent",
    "ambivalent": "Ambivalent",
    "ambiguous": "Ambivalent",
    "clear reply": "Clear Reply",
    "clear non-reply": "Clear Non-Reply",
    "clear nonreply": "Clear Non-Reply",
}

_EVASION_ALIASES = {
    "explicit": "Explicit",
    "implicit": "Implicit",
    "dodging": "Dodging",
    "general": "General",
    "deflection": "Deflection",
    "partial": "Partial/half-answer",
    "partial/half-answer": "Partial/half-answer",
    "partial answer": "Partial/half-answer",
    "half-answer": "Partial/half-answer",
    "declining to answer": "Declining to answer",
    "declining": "Declining to answer",
    "claims ignorance": "Claims ignorance",
    "ignorance": "Claims ignorance",
    "clarification": "Clarification",
}


def normalize_clarity(label: str) -> str:
    key = str(label).strip().lower()
    if key in _CLARITY_ALIASES:
        return _CLARITY_ALIASES[key]
    raise ValueError(f"Unknown clarity label: {label!r}")


def normalize_evasion(label: str) -> str | None:
    key = str(label).strip().lower()
    if key in _EVASION_ALIASES:
        return _EVASION_ALIASES[key]
    if key in ("", "nan", "none"):
        return None  # fine-grained label absent (e.g. official test split)
    raise ValueError(f"Unknown evasion label: {label!r}")


def clarity_from_evasion(evasion_label: str) -> str:
    return EVASION_TO_CLARITY[normalize_evasion(evasion_label)]


# ---------------------------------------------------------------------------
# QUD-relation signatures
# ---------------------------------------------------------------------------

class QUDRelation(str, Enum):
    """Relation between the *asked* question and the *addressed* question
    (the QUD reconstructed from the answer alone).

    EQUIVALENT       addressed == asked (same information request)
    SPECIFICATION    addressed is a strict sub-question of asked
                     (answers only one facet)
    GENERALIZATION   addressed is a less specific super-question of asked
    TOPIC_SHIFT      addressed is topically related but about a different
                     subject, agent, or time frame
    UNRELATED        addressed shares no substantive content with asked
    NONE             the answer addresses no QUD at all; it performs a
                     meta speech act (declining, claiming ignorance,
                     asking for clarification)
    """

    EQUIVALENT = "equivalent"
    SPECIFICATION = "specification"
    GENERALIZATION = "generalization"
    TOPIC_SHIFT = "topic_shift"
    UNRELATED = "unrelated"
    NONE = "none"


class SpeechAct(str, Enum):
    """Meta speech acts detectable when no QUD is addressed."""

    DECLINE = "decline"            # refuses to answer
    IGNORANCE = "ignorance"        # claims not to know
    CLARIFY = "clarify"            # asks the questioner for clarification
    ANSWER = "answer"              # a contentful answer attempt (default)


# Rule table: (best QUD relation, detected speech act) -> evasion label.
# Implicit vs Explicit is decided downstream by a directness check
# (does the answer state the requested information in the requested form?).
RELATION_RULES = {
    (QUDRelation.EQUIVALENT, SpeechAct.ANSWER): "Explicit",       # or Implicit (directness check)
    (QUDRelation.SPECIFICATION, SpeechAct.ANSWER): "Partial/half-answer",
    (QUDRelation.GENERALIZATION, SpeechAct.ANSWER): "General",
    (QUDRelation.TOPIC_SHIFT, SpeechAct.ANSWER): "Deflection",
    (QUDRelation.UNRELATED, SpeechAct.ANSWER): "Dodging",
    (QUDRelation.NONE, SpeechAct.DECLINE): "Declining to answer",
    (QUDRelation.NONE, SpeechAct.IGNORANCE): "Claims ignorance",
    (QUDRelation.NONE, SpeechAct.CLARIFY): "Clarification",
}
