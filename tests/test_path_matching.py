"""Tests for path_matches_glob — specifically that `**` matches zero or
more path segments, so a top-level test file is allowlisted.

Regression: `tests/**/*.py` previously rejected `tests/test_foo.py` (a
file directly under tests/) because fnmatch's `*` crosses `/` but the
literal `/` after `**` still required at least one intermediate segment.
The §3 test gate expects tests at `tests/test_*.py`, so top-level test
files must be allowed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import run  # noqa: E402


ALLOW = [g.format(package="vqasynth") for g in run.DEFAULT_ALLOWLIST_GLOBS]


def test_top_level_test_file_is_allowed():
    # The exact file that the VQASynth validation run wrongly rejected.
    assert run.path_matches_glob("tests/test_visual_degradation.py", ALLOW)


def test_nested_test_file_still_allowed():
    assert run.path_matches_glob("tests/sub/test_x.py", ALLOW)


def test_package_files_allowed_top_and_nested():
    assert run.path_matches_glob("vqasynth/foo.py", ALLOW)
    assert run.path_matches_glob("vqasynth/sub/bar.py", ALLOW)


def test_bundle_and_readme_allowed():
    assert run.path_matches_glob(".remyx-recommendation/SPEC.md", ALLOW)
    assert run.path_matches_glob("README.md", ALLOW)


def test_out_of_allowlist_paths_rejected():
    assert not run.path_matches_glob(".github/workflows/x.yml", ALLOW)
    assert not run.path_matches_glob("setup.py", ALLOW)
    assert not run.path_matches_glob("Dockerfile", ALLOW)


def test_always_blocked_still_matches():
    # The fix must not weaken the never-touch list.
    assert run.path_matches_glob(".github/workflows/ci.yml", run.ALWAYS_BLOCKED)
    assert run.path_matches_glob("docker/eval/Dockerfile", run.ALWAYS_BLOCKED)
    assert run.path_matches_glob("requirements.txt", run.ALWAYS_BLOCKED)
    assert run.path_matches_glob("pipelines/spatialvqa.yaml", run.ALWAYS_BLOCKED)
