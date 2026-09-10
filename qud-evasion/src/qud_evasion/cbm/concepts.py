"""Concept definitions for the bottleneck, as versioned data.

A concept is a yes / partly / no question the LLM answers about the TARGET
sub-question. `turn_aware` concepts refer to the sibling sub-questions of
the same interviewer turn; `has_which` concepts additionally return the
index of the sibling they refer to.

Adding a discovery round means adding a new key to CONCEPT_SETS. Nothing
downstream hardcodes concept ids.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Concept:
    id: str
    name: str
    question: str
    turn_aware: bool = False
    has_which: bool = False


V0: list[Concept] = [
    Concept(
        "c1", "stated_outright",
        "Is the information requested by the TARGET sub-question stated "
        "outright, in the form the question asks for (a yes/no question gets "
        "a yes or no, a when-question gets a time, a who-question gets a "
        "name, and so on)?",
    ),
    Concept(
        "c2", "inferable",
        "Can the requested information be inferred from the answer even "
        "though it is never stated in the requested form?",
    ),
    Concept(
        "c3", "partial",
        "Is only part of the TARGET sub-question addressed, with a facet or "
        "component left unanswered?",
    ),
    Concept(
        "c4", "general",
        "Is the TARGET sub-question answered only at a more general or vaguer "
        "level than it was asked, so that the specific thing requested is not "
        "resolved?",
    ),
    Concept(
        "c5", "redirect",
        "Does the answer redirect away from the TARGET sub-question to a "
        "different topic, person, event, or time frame?",
    ),
    Concept(
        "c6", "nonreply_act",
        "With respect to the TARGET sub-question, does the speaker refuse to "
        "answer, say they do not know, or ask for clarification instead of "
        "answering?",
    ),
    Concept(
        "c7", "acknowledge_then_move",
        "Does the speaker acknowledge the TARGET sub-question (for example by "
        "restating it, or saying they will come to it) and then move away "
        "from it without resolving it?",
    ),
    Concept(
        "c8", "addresses_sibling",
        "Does the answer address a DIFFERENT numbered sub-question from this "
        "same turn instead of the TARGET? If yes, give the number of that "
        "sub-question in \"which\".",
        turn_aware=True, has_which=True,
    ),
]

CONCEPT_SETS: dict[str, list[Concept]] = {
    "v0": V0,
}


def get_concepts(version: str = "v0") -> list[Concept]:
    try:
        return CONCEPT_SETS[version]
    except KeyError as e:
        raise ValueError(
            f"Unknown concept set {version!r}; known: {sorted(CONCEPT_SETS)}"
        ) from e