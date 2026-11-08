"""Sanity checks that must hold for every run. `python -m pytest tests` (or just run this file).
Set PROMO_TEST_SHOW to a folder of screenplays; defaults to the sample in tests/fixtures."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from promo_finder.llm import LLMConfig, parse_json  # noqa: E402
from promo_finder.parser import load_show  # noqa: E402
from promo_finder.pipeline import run  # noqa: E402

SHOW = os.getenv("PROMO_TEST_SHOW", str(Path(__file__).parent / "fixtures"))


def _run(provider="none", tmp="/tmp/promo_test"):
    return run(SHOW, tmp, LLMConfig(provider=provider, model="mock"))


def test_invariants():
    res = _run()
    eps = res["episodes"]
    for c in res["all"]:
        assert 20 <= c.length <= 60.01, (c.id, c.length)
        for s in c.segments:
            ep = eps[s.episode]
            cuts = {round(e.start, 2) for e in ep.events} | {round(e.end, 2) for e in ep.events}
            assert round(s.start, 2) in cuts and round(s.end, 2) in cuts, "timestamp not in script"
            for d in ep.events:      # never cut through a line of dialogue
                if d.kind == "DIALOGUE":
                    assert not (d.start < s.end - 0.3 and d.end > s.end + 0.3), (c.id, "splits line at end")
    ids = [c.id for c in res["post_first"]]
    assert len(ids) == len(set(ids))
    assert len({c.episode for c in res["post_first"]}) == len(ids), "one per episode in post-first"


def test_mock_llm_path():
    res = _run("mock", "/tmp/promo_test_mock")
    judged = [c for c in res["all"] if c.llm]
    assert judged, "LLM path produced no judgements"
    # the mock returns quotes that are NOT in the script: they must be flagged, not used
    assert all("mock hook" != c.hook for c in judged)
    assert any("quote not found" in f for c in judged for f in c.llm["flags"])


def test_parse_json_fenced():
    assert parse_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}


if __name__ == "__main__":
    test_invariants(); test_mock_llm_path(); test_parse_json_fenced(); print("all tests passed")
