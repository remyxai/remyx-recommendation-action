"""Unit tests for the candidate selection pass (lookback pool → pick the
most implementable candidate).

Covers the pure logic added alongside the selection pass: envelope→
Recommendation mapping, candidate brief rendering, select_recommendation's
parse / range / failure handling (which all fall back to candidates[0]),
and the PR-body selection section.

Run with: pytest tests/ -q
"""
import sys
from pathlib import Path

# run.py lives in src/ and isn't an installable package; put it on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import run  # noqa: E402
from run import Target  # noqa: E402


def _make_candidates():
    geo = run._paper_to_recommendation(
        {
            "title": "GeoWeaver",
            "resource_id": "2605.22558v1",
            "relevance_score": 0.98,
            "reasoning": "geometric grounding — a VLM architecture",
            "interest_name": "VQASynth",
            "resource": {"abstract": "Spatio-temporal reasoning in VLMs..."},
        },
        fallback_interest_name="fallback",
        interest_context="team focus body",
    )
    count = run._paper_to_recommendation(
        {
            "title": "HieraCount open-world counting",
            "resource_id": "2605.10887v1",
            "relevance_score": 0.87,
            "reasoning": "explicit counting granularity for prompt templates",
            "resource": {"abstract": "Open-world object counting remains brittle..."},
        },
        fallback_interest_name="VQASynth",
        interest_context="team focus body",
    )
    return geo, count


def test_paper_to_recommendation_maps_fields():
    geo, _ = _make_candidates()
    assert geo.paper_title == "GeoWeaver"
    assert geo.arxiv_id == "2605.22558v1"
    assert geo.relevance_score == 0.98
    assert geo.tier == "high"
    assert geo.interest_name == "VQASynth"          # from the paper envelope
    assert geo.interest_context == "team focus body"


def test_paper_to_recommendation_fallbacks():
    rec = run._paper_to_recommendation({"title": "X"}, "FB", "")
    assert rec.interest_name == "FB"                # falls back when absent
    assert rec.arxiv_id == ""
    assert rec.paper_title == "X"


def test_candidate_brief_is_indexed():
    geo, count = _make_candidates()
    brief = run._render_candidate_brief([geo, count])
    assert "[0] GeoWeaver" in brief
    assert "[1] HieraCount" in brief
    assert "relevance 0.98" in brief


def test_select_single_candidate_short_circuits(tmp_path):
    geo, _ = _make_candidates()
    # One candidate → no point spending a Claude call; returns None and the
    # caller uses candidates[0].
    assert run.select_recommendation(tmp_path, "pkg", [geo]) is None


def test_select_parses_chosen_index(tmp_path, monkeypatch):
    geo, count = _make_candidates()
    monkeypatch.setattr(
        run, "_run_claude_oneshot",
        lambda wd, p, t: (True, '{"chosen_index": 1, "reasoning": "clear call '
                                'site in prompts.py", "rejected": [{"index": 0, '
                                '"why": "model architecture, no call site"}]}'),
    )
    sel = run.select_recommendation(tmp_path, "pkg", [geo, count])
    assert sel is not None
    assert sel["chosen_index"] == 1                 # picked the lower-ranked, implementable one
    assert "call site" in sel["reasoning"]
    assert sel["rejected"][0]["index"] == 0


def test_select_out_of_range_falls_back(tmp_path, monkeypatch):
    geo, count = _make_candidates()
    monkeypatch.setattr(run, "_run_claude_oneshot",
                        lambda wd, p, t: (True, '{"chosen_index": 9}'))
    # Out-of-range → None → caller falls back to candidates[0].
    assert run.select_recommendation(tmp_path, "pkg", [geo, count]) is None


def test_select_non_int_index_falls_back(tmp_path, monkeypatch):
    geo, count = _make_candidates()
    monkeypatch.setattr(run, "_run_claude_oneshot",
                        lambda wd, p, t: (True, '{"chosen_index": "two"}'))
    assert run.select_recommendation(tmp_path, "pkg", [geo, count]) is None


def test_select_claude_failure_falls_back(tmp_path, monkeypatch):
    geo, count = _make_candidates()
    monkeypatch.setattr(run, "_run_claude_oneshot",
                        lambda wd, p, t: (False, "claude CLI timed out"))
    assert run.select_recommendation(tmp_path, "pkg", [geo, count]) is None


def test_select_unparseable_output_falls_back(tmp_path, monkeypatch):
    geo, count = _make_candidates()
    monkeypatch.setattr(run, "_run_claude_oneshot",
                        lambda wd, p, t: (True, "I think candidate 1 is best"))
    assert run.select_recommendation(tmp_path, "pkg", [geo, count]) is None


def test_pr_body_includes_selection_note_when_present():
    geo, _ = _make_candidates()
    tgt = Target(repo="remyxai/VQASynth", interest_id="x")
    body = run.build_pr_body(tgt, geo, True, "ok",
                             selection_note="picked for a clear call site in prompts.py")
    assert "Why this candidate" in body
    assert "clear call site" in body


def test_pr_body_suppresses_parenthetical_fallback_note():
    geo, _ = _make_candidates()
    tgt = Target(repo="remyxai/VQASynth", interest_id="x")
    # The fallback note ("(selection pass unavailable …)") starts with "(" and
    # is suppressed so the PR body doesn't show a non-explanation.
    body = run.build_pr_body(
        tgt, geo, True, "ok",
        selection_note="(selection pass unavailable — used top-ranked candidate)",
    )
    assert "Why this candidate" not in body


def test_pr_body_without_selection_note():
    geo, _ = _make_candidates()
    tgt = Target(repo="remyxai/VQASynth", interest_id="x")
    body = run.build_pr_body(tgt, geo, True, "ok")
    assert "Why this candidate" not in body
