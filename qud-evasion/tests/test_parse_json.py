from qud_evasion.qud.llm_client import parse_json


def test_plain_json():
    assert parse_json('{"a": 1}') == {"a": 1}


def test_fenced_json_with_prose():
    text = 'Sure! Here you go:\n```json\n{"relation": "topic_shift", "overlap": 0.3}\n```'
    assert parse_json(text)["relation"] == "topic_shift"


def test_garbage_returns_default():
    assert parse_json("no json here", default={"x": 0}) == {"x": 0}
