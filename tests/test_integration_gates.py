"""Tests for the role-based guardrails and the §2 invocation check.

Three behaviours, all introduced together:

1. ALWAYS_BLOCKED is role-based (filename/type), not directory-based, so
   `docker/` is no longer blanket-blocked — a Python stage driver under
   docker/ is editable (it's often the real call site), while Dockerfiles /
   shell scripts / dependency manifests stay blocked wherever they live.

2. check_integration requires INVOCATION: at least one newly-added
   function/method/class must be called from another changed file. An
   import alone no longer counts, and methods bolted onto an existing file
   that nothing calls are rejected (the shape that slipped through before).

3. changed_files uses --untracked-files=all so files inside a brand-new
   directory are seen per-file, not collapsed to the directory.

Run with: pytest tests/ -q
"""
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import run  # noqa: E402
from run import Target  # noqa: E402


# ── 1. role-based path policy ──────────────────────────────────────────────

ALLOW = [g.format(package="vqasynth") for g in run.DEFAULT_ALLOWLIST_GLOBS]


def _decision(path: str) -> str:
    if run.path_matches_glob(path, run.ALWAYS_BLOCKED):
        return "blocked"
    if run.path_matches_glob(path, ALLOW):
        return "allowed"
    return "rejected"


def test_docker_python_driver_is_editable():
    # The whole point: the call site under docker/ is no longer locked out.
    assert _decision("docker/eval_stage/process_eval.py") == "allowed"


def test_source_anywhere_is_editable():
    assert _decision("vqasynth/benchmarks.py") == "allowed"
    assert _decision("tests/test_x.py") == "allowed"
    assert _decision("scripts/run_thing.py") == "allowed"


def test_build_and_ci_files_blocked_by_role_anywhere():
    for p in [
        "Dockerfile",
        "docker/eval_stage/Dockerfile",
        "docker/base_image/Dockerfile.cpu",
        "docker/eval_stage/entrypoint.sh",
        "run.sh",
        "requirements.txt",
        "docker/eval_stage/requirements.txt",
        "setup.py",
        "pyproject.toml",
        "poetry.lock",
        ".github/workflows/ci.yml",
    ]:
        assert _decision(p) == "blocked", p


def test_github_block_overrides_python_allow():
    # A .py under .github must stay blocked even though *.py is allowlisted.
    assert _decision(".github/scripts/helper.py") == "blocked"


def test_prod_yaml_not_editable_via_allowlist():
    # Not blocked by directory anymore, but not allowlisted either.
    assert _decision("pipelines/spatialvqa.yaml") == "rejected"
    assert _decision("config/settings.yaml") == "rejected"


# ── 2. AST helpers + invocation check ──────────────────────────────────────

def test_public_callables_and_called_names():
    src = (
        "class Runner:\n"
        "    def score(self, x):\n        return x\n"
        "    def _private(self):\n        pass\n"
        "def top():\n    return Runner().score(1)\n"
    )
    assert run._public_callables(src) == {"Runner", "score", "top"}
    assert {"Runner", "score"} <= run._called_names(src)
    assert "_private" not in run._public_callables(src)


def _git(wd, *a):
    subprocess.run(["git", *a], cwd=wd, check=True, capture_output=True)


def _base_repo() -> Path:
    wd = Path(tempfile.mkdtemp())
    _git(wd, "init", "-q")
    _git(wd, "config", "user.email", "a@b.c")
    _git(wd, "config", "user.name", "t")
    (wd / "vqasynth").mkdir()
    (wd / "vqasynth" / "__init__.py").write_text("")
    (wd / "vqasynth" / "benchmarks.py").write_text(
        "class BenchmarkRunner:\n    def score(self, x):\n        return x\n"
    )
    (wd / "tests").mkdir()
    (wd / "tests" / "test_base.py").write_text("def test_base():\n    assert True\n")
    _git(wd, "add", "-A")
    _git(wd, "commit", "-qm", "base")
    return wd


TGT = Target(repo="r/x", interest_id="i")


def test_uncalled_new_method_is_rejected():
    wd = _base_repo()
    bp = wd / "vqasynth" / "benchmarks.py"
    bp.write_text(bp.read_text() + "    def run_spacedg(self, imgs):\n        return imgs\n")
    ok, violations = run.check_integration(wd, TGT, "vqasynth")
    assert not ok
    assert any("nothing calls" in v for v in violations)


def test_test_invocation_satisfies_integration():
    wd = _base_repo()
    bp = wd / "vqasynth" / "benchmarks.py"
    bp.write_text(bp.read_text() + "    def run_spacedg(self, imgs):\n        return imgs\n")
    (wd / "tests" / "test_spacedg.py").write_text(
        "from vqasynth.benchmarks import BenchmarkRunner\n"
        "def test_run():\n    BenchmarkRunner().run_spacedg([1])\n"
    )
    ok, _ = run.check_integration(wd, TGT, "vqasynth")
    assert ok


def test_import_without_call_is_rejected():
    wd = _base_repo()
    (wd / "vqasynth" / "newcap.py").write_text("def enhance(x):\n    return x * 2\n")
    (wd / "vqasynth" / "evaluation.py").write_text(
        "from vqasynth.newcap import enhance  # imported, never called\nVALUE = 1\n"
    )
    ok, violations = run.check_integration(wd, TGT, "vqasynth")
    assert not ok


def test_new_module_called_from_modified_file_passes():
    wd = _base_repo()
    (wd / "vqasynth" / "newcap.py").write_text("def enhance(x):\n    return x * 2\n")
    (wd / "vqasynth" / "evaluation.py").write_text(
        "from vqasynth.newcap import enhance\ndef go(x):\n    return enhance(x)\n"
    )
    ok, _ = run.check_integration(wd, TGT, "vqasynth")
    assert ok


def test_pure_edit_with_no_new_callable_is_not_gated():
    # Editing an existing function body (no new public callable) shouldn't
    # trip the invocation check.
    wd = _base_repo()
    bp = wd / "vqasynth" / "benchmarks.py"
    bp.write_text("class BenchmarkRunner:\n    def score(self, x):\n        return x + 1\n")
    ok, _ = run.check_integration(wd, TGT, "vqasynth")
    assert ok


# ── 3. changed_files sees files in brand-new directories ───────────────────

def test_changed_files_expands_new_directory():
    wd = _base_repo()
    (wd / "brand_new_dir").mkdir()
    (wd / "brand_new_dir" / "probe.py").write_text("x = 1\n")
    assert "brand_new_dir/probe.py" in run.changed_files(wd)
