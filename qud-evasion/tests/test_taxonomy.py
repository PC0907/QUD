"""Fast unit tests (no GPU, no network). Run with: pytest -q"""
import pandas as pd

from qud_evasion.data.taxonomy import (
    CLARITY_LABELS, EVASION_LABELS, EVASION_TO_CLARITY, QUDRelation,
    RELATION_RULES, SpeechAct, clarity_from_evasion, normalize_clarity,
    normalize_evasion,
)


def test_every_evasion_label_maps_to_a_clarity_label():
    for ev in EVASION_LABELS:
        assert EVASION_TO_CLARITY[ev] in CLARITY_LABELS


def test_alias_normalization():
    assert normalize_clarity("Ambivalent Reply") == "Ambivalent"
    assert normalize_evasion("partial") == "Partial/half-answer"
    try:
        normalize_clarity("totally honest")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_relation_rules_cover_all_speech_act_branches():
    for sa in (SpeechAct.DECLINE, SpeechAct.IGNORANCE, SpeechAct.CLARIFY):
        label = RELATION_RULES[(QUDRelation.NONE, sa)]
        assert clarity_from_evasion(label) == "Clear Non-Reply"


def test_relation_rules_answer_branch_clarity():
    assert clarity_from_evasion(
        RELATION_RULES[(QUDRelation.SPECIFICATION, SpeechAct.ANSWER)]
    ) == "Ambivalent"
    assert clarity_from_evasion(
        RELATION_RULES[(QUDRelation.EQUIVALENT, SpeechAct.ANSWER)]
    ) == "Clear Reply"
