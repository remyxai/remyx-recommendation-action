"""
run.py — Entry point for the remyxai/remyx-recommendation-action
composite GitHub Action.

The action runs once per workflow invocation; it opens a draft PR (or
an Issue when the recommended paper can't be cleanly scaffolded)
against the repo the action runs in.

Flow:

  1. Recommendation: GET /api/v1.0/papers/recommended on engine.remyx.ai
     for the configured ResearchInterest. Remyx server-side handles
     commit-history extraction, candidate pool, embedding pre-filter,
     and Gemini ranking — this action is a pure consumer.
  2. Confidence gate: skip Low / Noise tiers.
  3. Dedup: skip if an open PR already exists for this paper's arxiv_id
     (branch == `remyx-recommendation/{arxiv_id}`), or if any
     remyx-recommendation PR was opened within `rate-limit-days`.
  4. Clone the target repo (= GITHUB_REPOSITORY), branch from main.
  5. Write the spec bundle to `.remyx-recommendation/`:
       SPEC.md, PAPER.md, CONTEXT.md, GUARDRAILS.md, INVOCATION.md
  6. Invoke Claude Code (headless) with INVOCATION.md as the brief.
  7. Issue-fallback: if Claude wrote `.remyx-recommendation/OPEN_AS_ISSUE.md`
     (paper can't be scaffolded against this codebase) open an Issue
     with its reasoning and exit.
  8. Path-allowlist enforcement: reject if Claude touched files outside
     the allowed set.
  9. pytest in the workdir.
 10. Commit (with the bundle dir scrubbed), push, open the PR.

Inputs are read from env vars set by the action's `with:` block
(action.yml maps `inputs.X` → `INPUT_X`). Secrets and the workflow's
GITHUB_TOKEN are passed through unchanged.

  TARGET_REPO            — github.repository (the repo to operate on)
  INPUT_INTEREST_ID      — required, the Remyx ResearchInterest UUID
  INPUT_MIN_CONFIDENCE   — "high" | "moderate" | "low" (default: moderate)
  INPUT_DRAFT_MODE       — "always" | "on_test_failure" | "never" (default: always)
  INPUT_RATE_LIMIT_DAYS  — int, default 7
  REMYX_API_KEY          — engine.remyx.ai token (set as a workflow secret)
  ANTHROPIC_API_KEY      — Claude Code auth (set as a workflow secret)
  GITHUB_TOKEN           — workflow's built-in token, or a cross-repo PAT
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ─── Configuration ─────────────────────────────────────────────────────────

REMYX_API_BASE = os.environ.get("REMYX_API_BASE", "https://engine.remyx.ai")
REMYX_RECOMMENDATION_PERIOD = os.environ.get("REMYX_RECOMMENDATION_PERIOD", "week")
REMYX_RECOMMENDATION_LIMIT = int(os.environ.get("REMYX_RECOMMENDATION_LIMIT", "10"))

# Map Remyx's 0.0-1.0 relevance_score onto confidence-gate tiers.
# Thresholds are intentionally generous on the high end since the action
# is one-shot per run, not a ranked list — we just need a "should we
# open a PR for this?" gate.
RELEVANCE_TIER_FLOOR = {
    "high":     float(os.environ.get("REMYX_TIER_HIGH_FLOOR",     "0.80")),
    "moderate": float(os.environ.get("REMYX_TIER_MODERATE_FLOOR", "0.60")),
    "low":      float(os.environ.get("REMYX_TIER_LOW_FLOOR",      "0.40")),
}

TIER_RANK = {"high": 3, "moderate": 2, "low": 1, "noise": 0, "near-random": 0}

# Paths Claude Code is allowed to create/modify. Customers can extend
# via the `guardrails-allowlist` input on the action (comma-separated).
DEFAULT_ALLOWLIST_GLOBS = [
    "{package}/*_integration.py",
    "tests/test_*_integration.py",
    "tests/test_*.py",
    ".remyx-recommendation/**",
    "README.md",
]

BUNDLE_DIR_NAME = ".remyx-recommendation"
BRANCH_PREFIX = "remyx-recommendation/"
PR_TITLE_PREFIX = "[Remyx Recommendation]"

# Paths that are NEVER allowed to be touched (CI-affecting + dependency files)
ALWAYS_BLOCKED = [
    ".github/**",
    "docker/**",
    "pipelines/**",
    "config/**",
    "requirements.txt",
    "setup.py",
    "pyproject.toml",
    "MANIFEST.in",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("orchestrator")


# ─── Bundle + PR-body templates (module-level so they aren't indented by
# textwrap.dedent's "common leading whitespace" rule when interpolating
# multiline content like rec.spec_md or invocation prose). ────────────────

CANONICAL_ATTRIBUTION_URL = "https://engine.remyx.ai"
# Attribution links in PR bodies, Issues, and README appends point to
# the customer-facing product page on engine.remyx.ai, not to the
# orchestrator's private source repo (which would 404 for external
# readers).

# When Claude Code determines the paper can't be cleanly scaffolded (paper
# needs infra the repo lacks, integration point is too vague, datasets /
# checkpoints not available, etc.) it writes this file in the workdir
# instead of code. The orchestrator detects it and opens a discussion
# Issue rather than a PR — preserves the discovery surface without
# putting empty/throwaway scaffolding into a PR.
ISSUE_FALLBACK_FILENAME = f"{BUNDLE_DIR_NAME}/OPEN_AS_ISSUE.md"

_SPEC_MD_TEMPLATE = """\
# Implementation spec — drafted by Remyx Recommendation

**Recommended paper**: [{paper_title}](https://arxiv.org/abs/{arxiv_id})
**Confidence**: {tier} (Remyx relevance {relevance_score:.2f})
**Research interest**: {interest_name}

---

## Team's research focus

{interest_context_block}

## Why this paper for this team

{reasoning}

## Suggested experiment

{suggested_experiment}

## Paper abstract

{paper_abstract}
"""

_PAPER_MD_TEMPLATE = """\
# {paper_title}

arxiv: https://arxiv.org/abs/{arxiv_id}

## Abstract

{paper_abstract}
"""

_GUARDRAILS_MD_TEMPLATE = """\
# Path guardrails for this PR

You MAY create files matching:
```
{allowlist}
```

You MAY append-only modify:
```
README.md
```

You MUST NOT touch:
```
{blocked}
```

After the orchestrator validates your work, it checks the diff with
`git diff --name-only`. If any path you touched is outside the allowed
set, the PR is rejected and your work is not committed.
"""

_INVOCATION_MD_TEMPLATE = """\
You are a coding agent implementing a recommendation from the Remyx
Recommendation pipeline (attribution URL: {attribution_url}).

Read these files in order:
  1. .remyx-recommendation/SPEC.md       — the implementation spec (paper,
                                            why-this-paper, suggested
                                            experiment, team's research-
                                            focus body, abstract)
  2. .remyx-recommendation/PAPER.md      — paper title + abstract
  3. .remyx-recommendation/CONTEXT.md    — team context (recent merges,
                                            if Remyx returned any)
  4. .remyx-recommendation/GUARDRAILS.md — what you may and may not modify

Then look at the existing codebase structure (especially the `{package}/`
package and `tests/` directory) to understand the project's conventions
and what actually exists to integrate against.

# Step 1 — decide: PR or Issue

After reading the brief AND inspecting the relevant existing code, decide
whether you can produce a *concrete*, *non-vacuous* scaffold. Open as
ISSUE (not PR) if any of these is true:

  - The paper's contribution requires infrastructure the codebase lacks
    (e.g. a trainer when the repo is inference-only, a dataset format
    the repo never touches).
  - The integration point is too vague to pick a real module / API —
    you'd be inventing the integration rather than slotting into one.
  - The paper requires specific external checkpoints, datasets, or
    services the team clearly doesn't have access to, AND the scaffold
    couldn't usefully exist without them.
  - You searched the codebase for the relevant entry points and found
    nothing reasonable to extend or call into.

If ANY of the above hold, DO NOT WRITE CODE. Instead, write a file at
`{issue_fallback_filename}` with this exact shape (Markdown):

```
# Title: short, action-oriented (becomes the Issue title)
Optional one-line subtitle.

## Why this paper is interesting for the team

(2-3 sentences from the spec + your own reading)

## What blocks a clean implementation

(Specifics: missing infra, vague integration point, required external
artifacts, etc. Be concrete about what would need to exist for a real
integration to be drafted.)

## What we'd need to know / decide first

(1-3 questions or decisions the team should resolve before this becomes
implementable.)
```

The orchestrator detects this file and opens a GitHub Issue instead of a
draft PR. No code is committed, no PR is opened, no time is wasted on
scaffolding that would mislead a reviewer.

# Step 2 — only if you DIDN'T write the issue file: implement

Implement the MINIMAL-VIABLE-SCAFFOLDING version of the spec:

- Create one new module under `{package}/` (likely `{package}/<paper_slug>_integration.py`)
  with:
    * A config dataclass (e.g. `<Paper>Config`) holding the paper's reported
      hyperparameters as defaults
    * A class scaffold for the integration entry point. Keep heavy lifting
      (external checkpoint loading, etc.) as documented TODOs so this PR
      doesn't pretend to do work that requires external dependencies.
    * Any utility functions described in the spec (pixel conversions,
      data adapters, etc.) — implement these concretely.

- Create `tests/test_<paper_slug>_integration.py` with passing tests for
  every utility function you implemented concretely. Stub-test the class
  scaffold (smoke test of the no-checkpoint path returning sensible defaults).

- Append a brief "(Paper Title) Integration (experimental) 🧪" section
  to README.md. Attribute the work at the very end of the section
  with this exact line (one Markdown link to the customer-facing
  Remyx product page, no other URL):

      Contributed via [Remyx Recommendation]({attribution_url}).

  Do NOT use a different URL. The orchestrator's source repo is
  private; this link is the only one that resolves for external readers.

Run pytest before declaring done. If tests fail, fix them or scope your
implementation down until they pass. Do not modify files outside the
guardrails allowlist.

# CRITICAL: do not run git commands

You MUST NOT run any `git` command during your session (`git init`,
`git checkout`, `git stash`, `git reset`, `git commit`, `git add`,
`git rm`, `git rebase`, etc. — none of them). The orchestrator manages
all version control. Even commands you think are read-only (e.g.
`git status`) are forbidden — past runs have hit subtle issues where
agents ran a `git checkout` to back out a half-edit and left the
working tree in an orphan state that broke the PR.

If you need to back out an edit, use the file-edit tools to restore
the file's content. If you're unsure what the original content was,
look it up via the standard read tools — do not invoke git.

When complete, output a one-paragraph SUMMARY of what you actually built.
Be honest about what you stubbed vs implemented.
"""

_PR_BODY_TEMPLATE = """\
> **Drafted by an autonomous discovery loop** — Remyx ranks recent arXiv papers against this team's research interest and shipping history; Claude Code implements the top pick.
>
> **Recommended paper**: [{paper_title}](https://arxiv.org/abs/{arxiv_id})
> **Confidence**: {tier_emoji} {tier} (Remyx relevance {relevance_score:.2f})
> **Research interest**: {interest_name}
> **Implementation by**: Claude Code as autonomous agent

---

## Why this paper for this team

{reasoning}

## Suggested experiment

{suggested_experiment}

---

{test_section}

---

_Opened by the [Remyx Recommendation]({attribution_url}) orchestrator._
"""


# ─── Data classes ──────────────────────────────────────────────────────────


DRAFT_MODES = ("always", "on_test_failure", "never")


@dataclass
class Target:
    repo: str                         # "owner/name" — the target repo
                                      # (where PRs and Issues land).
                                      # The action either runs in this repo
                                      # (same-repo customer install) or
                                      # operates on it cross-repo from a
                                      # controller repo with a PAT in
                                      # FF_GITHUB_TOKEN. There is no
                                      # "fork mode" — PRs always go to
                                      # `repo` directly.
    interest_id: str = ""             # Remyx ResearchInterest UUID — pre-filled
                                      # from the engine.remyx.ai workflow snippet
    min_confidence: str = "moderate"
    rate_limit_days: int = 7
    # PR-draft policy:
    #   "always"          — every PR opens as draft (default; future
    #                       webhook + Modal eval flow will mark them
    #                       ready after the team's own evals pass)
    #   "on_test_failure" — tests pass: ready; tests fail: draft
    #   "never"           — tests pass: ready; tests fail: SKIP (don't open
    #                       PR at all). Equivalent to the old
    #                       draft_on_test_failure=False behavior.
    draft_mode: str = "always"
    guardrails_allowlist: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class Recommendation:
    paper_title: str
    arxiv_id: str
    tier: str                         # "high" / "moderate" / "low" / "noise"
    z_score: float                    # legacy; unused since the Remyx-API pivot
    spec_md: str                      # legacy; PR body now sources from
                                      # reasoning + suggested_experiment instead
    paper_abstract: str
    team_context: str
    domain_summary: str
    raw_paper_md: str
    # New fields populated by query_remyx_recommendation() — match the Remyx
    # /papers/recommended response envelope so downstream renderers can pull
    # whichever fields they need.
    relevance_score: float = 0.0
    reasoning: str = ""
    suggested_experiment: str = ""
    recommendation_id: str = ""
    interest_name: str = ""
    interest_context: str = ""        # rich text body the customer wrote
                                      # on engine.remyx.ai (research focus,
                                      # current goals, what they care about)


# ─── Helpers ───────────────────────────────────────────────────────────────


def _github_token() -> str:
    """Resolve the GitHub token to use for git push + API calls.

    Preference order:
      1. INPUT_GITHUB_TOKEN — explicit cross-repo PAT override
      2. GITHUB_TOKEN — the workflow's built-in token (action.yml's
         step env sets this from `${{ github.token }}`).

    Two separate env vars rather than a single `${{ a || b }}` in
    action.yml because GitHub Actions' || operator on empty-string
    inputs returns '' instead of falling through (observed via v1.0.3
    git-push failure). Resolving in Python gives reliable semantics.
    """
    return (
        os.environ.get("INPUT_GITHUB_TOKEN", "").strip()
        or os.environ.get("GITHUB_TOKEN", "").strip()
    )


def gh_api(method: str, path: str, body: dict | None = None) -> Any:
    """Minimal GitHub API wrapper."""
    token = _github_token()
    if not token:
        raise RuntimeError(
            "Neither INPUT_GITHUB_TOKEN nor GITHUB_TOKEN is set. The "
            "action.yml should pass ${{ github.token }} as GITHUB_TOKEN "
            "by default; if you're invoking the script outside an Action, "
            "export GITHUB_TOKEN manually."
        )
    url = f"https://api.github.com{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "feature-finder-orchestrator",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"GitHub {method} {path} → HTTP {e.code}: {body_text}") from e


def slugify(s: str, max_len: int = 60) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s.lower()).strip("-")
    return s[:max_len]


def strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "").strip()


# ─── Remyx API recommendation ──────────────────────────────────────────────


def _remyx_get(path: str, *, params: dict | None = None) -> dict:
    """GET against the Remyx engine API with the configured API key.
    Raises RuntimeError on non-2xx response."""
    api_key = os.environ.get("REMYX_API_KEY") or os.environ.get("REMYXAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "REMYX_API_KEY (or REMYXAI_API_KEY) is required. Generate one "
            "from your engine.remyx.ai settings and add it as a workflow "
            "secret."
        )
    url = REMYX_API_BASE.rstrip("/") + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "feature-finder-orchestrator",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:400]
        raise RuntimeError(
            f"Remyx API GET {path} → HTTP {e.code}: {body}"
        ) from e


def _relevance_to_tier(score: float) -> str:
    if score >= RELEVANCE_TIER_FLOOR["high"]:
        return "high"
    if score >= RELEVANCE_TIER_FLOOR["moderate"]:
        return "moderate"
    if score >= RELEVANCE_TIER_FLOOR["low"]:
        return "low"
    return "noise"


def query_remyx_recommendation(target: Target) -> Recommendation:
    """Pull the top recommendation for ``target.interest_id`` from the
    Remyx engine. Replaces the previous Gemini-direct path — Remyx now
    owns commit-history extraction, candidate pool, embedding pre-filter,
    Gemini ranking, and reasoning generation. The action is a pure
    consumer.

    See ``GET /api/v1.0/papers/recommended`` in remyxai/remyx
    (engine/app/api/papers.py).
    """
    if not target.interest_id:
        raise RuntimeError(
            f"target {target.repo!r} has no interest_id configured. "
            f"Get the interest_id from engine.remyx.ai (Settings → "
            f"Workflow snippet) and pass it via the action's "
            f"`with: interest-id: ...` input."
        )

    log.info(f"  → querying Remyx /papers/recommended "
             f"(interest={target.interest_id[:8]}…)")
    resp = _remyx_get(
        "/api/v1.0/papers/recommended",
        params={
            "interest_id": target.interest_id,
            "period":      REMYX_RECOMMENDATION_PERIOD,
            "limit":       REMYX_RECOMMENDATION_LIMIT,
        },
    )
    papers = resp.get("papers") or []
    if not papers:
        raise RuntimeError(
            f"Remyx returned no recommendations for interest "
            f"{target.interest_id} in period={REMYX_RECOMMENDATION_PERIOD}. "
            f"Either the interest has no fresh picks, or the daily refresh "
            f"hasn't run since the last cron. Try POSTing to "
            f"/api/v1.0/papers/recommended/refresh first."
        )

    # Pick the highest-scoring recommendation that isn't already in flight.
    # Existing-PR dedup runs LATER in process_target — here we just take the
    # top of the list and let the dedup gate skip if needed.
    top = papers[0]
    relevance = float(top.get("relevance_score") or 0.0)
    tier = _relevance_to_tier(relevance)

    resource = top.get("resource") or {}
    arxiv_id = top.get("resource_id") or resource.get("arxiv_id") or ""
    abstract = (resource.get("abstract") or resource.get("summary") or "").strip()

    # Fetch the interest's full context body — the rich text the customer
    # wrote on engine.remyx.ai about their research focus / goals. This
    # gives Claude Code a far better picture of what to build than the
    # paper abstract + recommendation reasoning alone.
    interest_name = top.get("interest_name") or ""
    interest_context = ""
    try:
        interest = _remyx_get(f"/api/v1.0/research-interests/{target.interest_id}")
        interest_context = (interest.get("context") or "").strip()
        if not interest_name:
            interest_name = interest.get("name") or ""
    except Exception as e:
        log.warning(f"    (interest context fetch failed: {e}; "
                    f"continuing with reasoning-only brief)")

    log.info(f"    ✓ {top.get('title','?')[:60]}…  "
             f"relevance={relevance:.2f}  tier={tier}")

    return Recommendation(
        paper_title=top.get("title") or "(untitled)",
        arxiv_id=arxiv_id,
        tier=tier,
        z_score=0.0,                       # legacy field, unused
        spec_md="",                        # legacy; rendered from fields below
        paper_abstract=abstract,
        team_context="",                   # Remyx keeps team context server-side
        domain_summary="",
        raw_paper_md="",
        relevance_score=relevance,
        reasoning=(top.get("reasoning") or "").strip(),
        suggested_experiment=(top.get("suggested_experiment") or "").strip(),
        recommendation_id=top.get("recommendation_id") or "",
        interest_name=interest_name,
        interest_context=interest_context,
    )



# ─── Dedup ─────────────────────────────────────────────────────────────────


def existing_pr_for(target: Target, branch: str) -> dict | None:
    """Return the PR dict if an open PR exists on the target repo for `branch`."""
    head_owner = target.repo.split("/")[0]
    head = f"{head_owner}:{branch}"
    prs = gh_api("GET", f"/repos/{target.repo}/pulls?state=open&head={head}")
    return prs[0] if prs else None


def recent_pr_within_rate_limit(target: Target) -> bool:
    """Return True if a Remyx Recommendation PR was opened on the target
    repo within `rate_limit_days`."""
    if target.rate_limit_days <= 0:
        return False
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=target.rate_limit_days)
    prs = gh_api(
        "GET", f"/repos/{target.repo}/pulls?state=all&per_page=20"
    )
    for pr in prs:
        ref = pr.get("head", {}).get("ref", "")
        if not ref.startswith(BRANCH_PREFIX):
            continue
        created = dt.datetime.fromisoformat(pr["created_at"].replace("Z", "+00:00"))
        if created > cutoff:
            log.info(
                f"  rate-limit hit: {pr['html_url']} opened "
                f"{(dt.datetime.now(dt.timezone.utc) - created).days}d ago"
            )
            return True
    return False


# ─── Workdir + spec bundle ─────────────────────────────────────────────────


def prepare_workdir(target: Target) -> Path:
    """Clone the target repo, return the workdir.

    The action operates on `target.repo` directly — branches are pushed
    to it, PRs open against its main. Authentication is via
    GITHUB_TOKEN (either the workflow's built-in token when the action
    runs in the target repo, or a cross-repo PAT like FF_GITHUB_TOKEN
    when the action lives in a separate controller repo).
    """
    workdir = Path(tempfile.mkdtemp(prefix=f"rr-{slugify(target.repo)}-"))
    token = _github_token()
    if not token:
        raise RuntimeError(
            "No GitHub token available for clone+push. Either pass "
            "`with: github-token: ${{ secrets.MY_PAT }}` or rely on the "
            "default ${{ github.token }} the action.yml threads through."
        )
    # Use the modern github.com auth convention: token as the `x-access-token`
    # user. This is more portable across the workflow GITHUB_TOKEN (which
    # works fine with the bare-token-as-username form too) and PATs (which
    # work either way), avoiding any ambiguity that left the clone URL
    # credential-less on the v1.0.3 push failure.
    repo_url = f"https://x-access-token:{token}@github.com/{target.repo}.git"

    log.info(f"  → cloning {target.repo} to {workdir}")
    subprocess.run(["git", "clone", "--depth", "20", repo_url, str(workdir)], check=True)
    subprocess.run(
        ["git", "config", "user.email", "remyx-recommendation@noreply.remyx.ai"],
        cwd=workdir, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Remyx Recommendation"],
        cwd=workdir, check=True,
    )
    return workdir


def detect_package_name(workdir: Path) -> str:
    """Best-effort guess at the importable package name in workdir."""
    for cand in workdir.iterdir():
        if cand.is_dir() and (cand / "__init__.py").exists() and not cand.name.startswith((".", "test")):
            return cand.name
    return "src"


def write_spec_bundle(workdir: Path, target: Target, rec: Recommendation, package: str) -> None:
    """Write the .remyx-recommendation/ bundle that Claude Code reads as its brief."""
    bundle = workdir / BUNDLE_DIR_NAME
    bundle.mkdir(exist_ok=True)

    interest_block = (
        rec.interest_context
        if rec.interest_context
        else "(no research-focus body configured for this interest on engine.remyx.ai)"
    )
    (bundle / "SPEC.md").write_text(_SPEC_MD_TEMPLATE.format(
        paper_title=rec.paper_title,
        arxiv_id=rec.arxiv_id,
        tier=rec.tier,
        relevance_score=rec.relevance_score,
        interest_name=rec.interest_name or "(unnamed interest)",
        interest_context_block=interest_block,
        reasoning=rec.reasoning or "(no reasoning provided)",
        suggested_experiment=rec.suggested_experiment or "(none)",
        paper_abstract=rec.paper_abstract or "(abstract unavailable)",
    ))

    (bundle / "PAPER.md").write_text(_PAPER_MD_TEMPLATE.format(
        paper_title=rec.paper_title,
        arxiv_id=rec.arxiv_id,
        paper_abstract=rec.paper_abstract,
    ))

    if rec.team_context:
        (bundle / "CONTEXT.md").write_text(
            f"# Team context (Gemini-extracted from {target.repo} merge history)\n\n"
            f"{rec.team_context}\n"
        )

    allowlist = target.guardrails_allowlist or [
        g.format(package=package) for g in DEFAULT_ALLOWLIST_GLOBS
    ]
    (bundle / "GUARDRAILS.md").write_text(_GUARDRAILS_MD_TEMPLATE.format(
        allowlist="\n".join(allowlist),
        blocked="\n".join(ALWAYS_BLOCKED),
    ))

    (bundle / "INVOCATION.md").write_text(_INVOCATION_MD_TEMPLATE.format(
        package=package,
        attribution_url=CANONICAL_ATTRIBUTION_URL,
        issue_fallback_filename=ISSUE_FALLBACK_FILENAME,
    ))


# ─── Claude Code invocation ────────────────────────────────────────────────


def invoke_claude_code(workdir: Path, timeout_s: int = 600) -> tuple[bool, str]:
    """Invoke the Claude Code CLI in headless mode with the workdir as context.

    Returns (success, stdout/stderr). Success means CLI exit 0 — caller still
    validates the produced changes with the path-allowlist check + tests.
    """
    invocation = (workdir / BUNDLE_DIR_NAME / "INVOCATION.md").read_text()
    log.info(f"  → invoking Claude Code (timeout={timeout_s}s) in {workdir}")
    try:
        # `claude` CLI: -p for prompt, --dangerously-skip-permissions for CI
        result = subprocess.run(
            [
                "claude",
                "--dangerously-skip-permissions",
                "-p", invocation,
            ],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        ok = result.returncode == 0
        output = (result.stdout or "") + ("\n--- STDERR ---\n" + result.stderr if result.stderr else "")
        return ok, output[-4000:]   # last 4KB for log brevity
    except subprocess.TimeoutExpired:
        return False, f"claude CLI timed out after {timeout_s}s"
    except FileNotFoundError:
        return False, "claude CLI not found on PATH (install: npm install -g @anthropic-ai/claude-code)"


# ─── Validation ────────────────────────────────────────────────────────────


# Build-artifact paths that show up in `git status` as side-effects of
# running tests / imports during the Claude Code session — not intentional
# changes. Filtered out before the allowlist check.
_BUILD_ARTIFACT_SUBSTRINGS = (
    "__pycache__/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".coverage",
)
_BUILD_ARTIFACT_SUFFIXES = (".pyc", ".pyo")


def changed_files(workdir: Path) -> list[str]:
    """Files Claude Code modified or created (vs HEAD), excluding build-
    artifact side-effects (__pycache__, .pytest_cache, *.pyc, etc.).

    Without this filter, pytest's bytecode cache shows up as 'untracked'
    files in git status and gets the run rejected for path-allowlist
    violations even though Claude never intentionally wrote them."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=workdir, capture_output=True, text=True, check=True,
    )
    paths = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        # Format: "XY path/to/file" — XY is status flags
        p = line[3:].strip()
        # Status output can quote paths that contain spaces; strip the quotes.
        if p.startswith('"') and p.endswith('"'):
            p = p[1:-1]
        if any(sub in p for sub in _BUILD_ARTIFACT_SUBSTRINGS):
            continue
        if any(p.endswith(suf) for suf in _BUILD_ARTIFACT_SUFFIXES):
            continue
        paths.append(p)
    return paths


def path_matches_glob(path: str, patterns: list[str]) -> bool:
    """Simple glob matcher: ** matches any path, * matches one segment."""
    import fnmatch
    return any(fnmatch.fnmatch(path, p) or fnmatch.fnmatch(path, p.replace("**", "*"))
               for p in patterns)


def validate_changes(workdir: Path, target: Target, package: str) -> tuple[bool, list[str]]:
    """Returns (passed_allowlist, violations)."""
    allowlist = target.guardrails_allowlist or [g.format(package=package) for g in DEFAULT_ALLOWLIST_GLOBS]
    paths = changed_files(workdir)
    violations = []
    for p in paths:
        if path_matches_glob(p, ALWAYS_BLOCKED):
            violations.append(f"BLOCKED: {p}")
            continue
        if not path_matches_glob(p, allowlist):
            violations.append(f"NOT IN ALLOWLIST: {p}")
    return (not violations, violations)


def run_tests(workdir: Path, timeout_s: int = 300) -> tuple[bool, str]:
    """Run pytest. Returns (passed, output)."""
    log.info(f"  → running pytest in {workdir}")
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", "-q", "--maxfail=3"],
            cwd=workdir, capture_output=True, text=True, timeout=timeout_s,
        )
        ok = result.returncode == 0
        output = (result.stdout or "") + ("\n--- STDERR ---\n" + result.stderr if result.stderr else "")
        return ok, output[-3000:]
    except subprocess.TimeoutExpired:
        return False, f"pytest timed out after {timeout_s}s"
    except Exception as e:
        return False, f"pytest invocation failed: {e}"


# ─── PR opening ────────────────────────────────────────────────────────────


def open_pr(target: Target, branch: str, title: str, body: str, draft: bool) -> str:
    """Open a PR on the target repo; returns the PR URL."""
    log.info(f"  → opening {'draft' if draft else ''} PR on {target.repo}")
    pr = gh_api("POST", f"/repos/{target.repo}/pulls", {
        "title": title,
        "head": branch,
        "base": "main",
        "body": body,
        "draft": draft,
    })
    return pr["html_url"]


def open_issue(target: Target, title: str, body: str) -> str:
    """Open a discussion Issue on the target repo. Returns the issue URL."""
    full_body = (
        f"{body}\n\n---\n\n"
        f"_Opened by the [Remyx Recommendation]({CANONICAL_ATTRIBUTION_URL}) "
        f"orchestrator — no PR was opened because the orchestrator's coding "
        f"agent determined the paper couldn't be cleanly scaffolded against "
        f"the current codebase._"
    )
    log.info(f"  → opening Issue on {target.repo}")
    issue = gh_api("POST", f"/repos/{target.repo}/issues", {
        "title": title,
        "body": full_body,
    })
    return issue["html_url"]


def parse_issue_fallback_file(path: Path) -> tuple[str, str]:
    """Parse Claude's OPEN_AS_ISSUE.md into (title, body). The expected
    shape is:

        # Title: short description
        (optional subtitle)

        ## Why this paper is interesting ...

    First H1 (with optional 'Title:' prefix) becomes the Issue title;
    everything after is the body. Falls back to a generic title if no
    H1 is found."""
    text = path.read_text().strip()
    lines = text.splitlines()
    title = ""
    body_start = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("# "):
            inner = s[2:].strip()
            if inner.lower().startswith("title:"):
                inner = inner[len("title:"):].strip()
            title = inner
            body_start = i + 1
            break
    if not title:
        title = "Remyx Recommendation: paper needs team discussion"
    body = "\n".join(lines[body_start:]).strip()
    return title, body


def commit_and_push(workdir: Path, branch: str, title: str) -> None:
    """Stage all changes, commit, and push the branch to origin.

    Two classes of files are scrubbed before staging so they don't end
    up in the PR even when the target repo's .gitignore doesn't cover
    them:

      - Build-artifact directories (__pycache__, .pytest_cache,
        .mypy_cache, .ruff_cache) — side-effects of running tests /
        imports during the Claude session.
      - The orchestrator's own bundle directory (.remyx-recommendation)
        — these are briefing material the action wrote for Claude to
        read; SPEC.md / PAPER.md / GUARDRAILS.md / INVOCATION.md
        duplicate content already in the PR body and add noise to the
        diff.
    """
    # Sanity check: make sure local HEAD still equals origin/main before
    # we branch. If Claude (or pytest) disturbed the git state during the
    # session — `git checkout --orphan`, `rm -rf .git`, `git init`,
    # whatever — local main can diverge from remote main, and the
    # subsequent `git checkout -b branch` produces a root-commit branch
    # with no history in common with main. The PR-creation API then
    # rejects with HTTP 422. Fail fast with a clear error instead.
    head_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workdir, capture_output=True, text=True, check=True,
    ).stdout.strip()
    try:
        remote_sha = subprocess.run(
            ["git", "rev-parse", "origin/main"],
            cwd=workdir, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        remote_sha = ""
    if not remote_sha or head_sha != remote_sha:
        raise RuntimeError(
            f"local HEAD ({head_sha[:8]}) doesn't match origin/main "
            f"({(remote_sha or 'MISSING')[:8]}) — git state was disturbed "
            f"during the session. Refusing to commit; would produce a "
            f"root-commit branch and fail at PR creation."
        )

    subprocess.run(["git", "checkout", "-b", branch], cwd=workdir, check=True)

    # Scrub build artifacts (pytest bytecode caches, mypy/ruff caches).
    # IMPORTANT: prune .git/ from the traversal. Branch names that
    # contain `/` create directories under .git/refs/heads/, and we
    # name our branches `remyx-recommendation/<arxiv_id>`. Any pattern
    # that happens to match a directory name inside .git/ would let
    # `rm -rf` wipe a branch ref and produce an orphan root-commit —
    # which then 422s at PR creation with "no history in common with
    # main." Pruning .git/ is the load-bearing safety here.
    for pat in ("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"):
        subprocess.run(
            ["find", ".", "-path", "./.git", "-prune",
             "-o", "-type", "d", "-name", pat,
             "-exec", "rm", "-rf", "{}", "+"],
            cwd=workdir, check=False,
        )

    # Bundle dir is always at the top level; remove explicitly so we
    # never have to walk into it with find. The bundle files were
    # briefing material for Claude (SPEC.md, INVOCATION.md, etc.) —
    # they duplicate the PR body and have no business in the commit.
    bundle_path = workdir / BUNDLE_DIR_NAME
    if bundle_path.exists():
        shutil.rmtree(bundle_path, ignore_errors=True)
    subprocess.run(["git", "add", "-A"], cwd=workdir, check=True)
    subprocess.run(["git", "commit", "-m", title], cwd=workdir, check=True)

    # Delete any orphan branch with the same name from the remote before
    # pushing. Two reasons:
    #   1. The existing-PR dedup gate already skipped if an OPEN PR for
    #      this branch exists. By the time we get here, any remote branch
    #      with the same name is from a CLOSED PR and is safe to remove.
    #   2. `--force` push from a shallow clone (we use --depth 20)
    #      confuses GitHub's PR validator — it treats the pushed branch
    #      as rooted ("no history in common with main") and refuses PR
    #      creation. Delete-then-plain-push avoids the force entirely.
    # `check=False` because a non-existent branch is the common case and
    # the delete is a no-op there.
    subprocess.run(
        ["git", "push", "origin", "--delete", branch],
        cwd=workdir, check=False, capture_output=True,
    )
    subprocess.run(
        ["git", "push", "-u", "origin", branch],
        cwd=workdir, check=True,
    )


# ─── Main per-target loop ──────────────────────────────────────────────────


def process_target(target: Target) -> dict:
    """Run the full discovery + implementation loop for one target.
    Returns a status dict suitable for logging / Slack notify."""
    result: dict = {"repo": target.repo, "status": "unknown"}

    # 1+2. Query + gate
    rec = query_remyx_recommendation(target)
    result.update({
        "paper": rec.paper_title,
        "arxiv": rec.arxiv_id,
        "tier": rec.tier,
    })
    log.info(f"  ✓ recommendation: [{rec.tier}] {rec.paper_title}")

    min_required = TIER_RANK.get(target.min_confidence.lower(), 2)
    actual = TIER_RANK.get(rec.tier.lower(), 0)
    if actual < min_required:
        result["status"] = "skipped_low_confidence"
        log.info(f"  ✗ tier {rec.tier} below min {target.min_confidence}; skipping")
        return result

    # 3. Dedup
    if recent_pr_within_rate_limit(target):
        result["status"] = "skipped_rate_limit"
        return result

    branch = f"{BRANCH_PREFIX}{rec.arxiv_id or slugify(rec.paper_title)}"
    if existing_pr_for(target, branch):
        result["status"] = "skipped_pr_exists"
        log.info(f"  ✗ PR already exists for branch {branch}; skipping")
        return result

    # 4-5. Workdir + spec bundle
    workdir = prepare_workdir(target)
    try:
        package = detect_package_name(workdir)
        log.info(f"  detected package: {package}")
        write_spec_bundle(workdir, target, rec, package)

        # 6. Claude Code
        ok, claude_log = invoke_claude_code(workdir)
        result["claude_exit_ok"] = ok
        # Always retain the Claude log tail — useful for diagnosing
        # silent-success-but-broken-state outcomes (e.g. orphan branch,
        # missing files), not just hard failures.
        result["claude_log_tail"] = claude_log[-1000:]
        if not ok:
            result["status"] = "claude_failed"
            return result

        # 6.5. Claude may have elected Issue-mode instead of writing code
        # (paper can't be cleanly scaffolded against this codebase; spec
        # too vague; needed infra missing; etc.). When it does, the brief
        # told it to write OPEN_AS_ISSUE.md INSTEAD of any code, so no
        # PR makes sense — open an Issue with its reasoning.
        issue_file = workdir / ISSUE_FALLBACK_FILENAME
        if issue_file.exists():
            log.info(f"  → Claude elected Issue-mode "
                     f"({ISSUE_FALLBACK_FILENAME} present); opening Issue")
            issue_title_inner, issue_body_inner = parse_issue_fallback_file(issue_file)
            issue_title = f"[Remyx Recommendation] {issue_title_inner}"
            issue_body = (
                f"**Recommended paper**: "
                f"[{rec.paper_title}](https://arxiv.org/abs/{rec.arxiv_id})\n"
                f"**Confidence**: {rec.tier} "
                f"(Remyx relevance {rec.relevance_score:.2f})\n"
                f"**Research interest**: {rec.interest_name or '(unnamed)'}\n"
                f"\n---\n\n"
                f"{issue_body_inner}"
            )
            issue_url = open_issue(target, issue_title, issue_body)
            result["status"] = "issue_opened"
            result["issue_url"] = issue_url
            log.info(f"  ✓ issue_opened: {issue_url}")
            return result

        # 7. Path allowlist enforcement
        passed_allowlist, violations = validate_changes(workdir, target, package)
        if not passed_allowlist:
            result["status"] = "rejected_path_violations"
            result["violations"] = violations
            log.warning(f"  ✗ path violations: {violations}")
            return result

        # 8. Tests
        tests_passed, test_output = run_tests(workdir)
        result["tests_passed"] = tests_passed
        if target.draft_mode == "always":
            draft = True
        elif target.draft_mode == "never":
            if not tests_passed:
                result["status"] = "skipped_test_failure"
                result["test_output_tail"] = test_output[-500:]
                return result
            draft = False
        else:                                # "on_test_failure"
            draft = not tests_passed

        # 9. Commit + push + PR
        pr_title = f"{PR_TITLE_PREFIX} {rec.paper_title}"
        pr_body = build_pr_body(target, rec, tests_passed, test_output)
        commit_and_push(workdir, branch, pr_title)
        pr_url = open_pr(target, branch, pr_title, pr_body, draft=draft)
        result["status"] = "pr_opened_draft" if draft else "pr_opened"
        result["pr_url"] = pr_url
        log.info(f"  ✓ {result['status']}: {pr_url}")
        return result

    finally:
        # Clean up tmpdir unless DEBUG_KEEP_WORKDIR set
        if not os.environ.get("DEBUG_KEEP_WORKDIR"):
            shutil.rmtree(workdir, ignore_errors=True)


def build_pr_body(target: Target, rec: Recommendation, tests_passed: bool, test_output: str) -> str:
    tier_emoji = {"high": "🟢", "moderate": "🟡", "low": "🟠", "noise": "🔴"}.get(rec.tier, "⚪")
    test_section = (
        "### Test results\n\n✅ All tests passed.\n"
        if tests_passed else
        f"### Test results\n\n⚠️ Tests did not pass. PR opened as draft for review.\n\n```\n{test_output[-1000:]}\n```\n"
    )
    return _PR_BODY_TEMPLATE.format(
        paper_title=rec.paper_title,
        arxiv_id=rec.arxiv_id,
        tier_emoji=tier_emoji,
        tier=rec.tier,
        relevance_score=rec.relevance_score,
        interest_name=rec.interest_name or "(unnamed)",
        reasoning=rec.reasoning or "(no reasoning provided)",
        suggested_experiment=rec.suggested_experiment or "(none)",
        test_section=test_section,
        attribution_url=CANONICAL_ATTRIBUTION_URL,
    )


# ─── Entry point ───────────────────────────────────────────────────────────


def _require_env(name: str) -> str:
    """Read a required env var or exit with a clear error."""
    v = os.environ.get(name, "").strip()
    if not v:
        log.error(
            f"Required env var {name!r} is empty or unset. "
            f"Check the action's `with:` block (for INPUT_* vars) or "
            f"the workflow's `env:` / `secrets:` block (for "
            f"REMYX_API_KEY, ANTHROPIC_API_KEY, GITHUB_TOKEN)."
        )
        sys.exit(2)
    return v


def _optional_env(name: str, default: str) -> str:
    return (os.environ.get(name) or "").strip() or default


def build_target_from_env() -> Target:
    """Read the action inputs from env vars and build a single Target.

    GitHub Actions composite actions surface `inputs.foo` as the env
    var `INPUT_FOO` to subprocesses (the case is normalized to upper
    when passing through; action.yml is responsible for the mapping).
    The action's `runs.steps` block sets these explicitly to be
    portable across composite / Docker / JavaScript action types.
    """
    repo = _require_env("TARGET_REPO")
    interest_id = _require_env("INPUT_INTEREST_ID")

    draft_mode = _optional_env("INPUT_DRAFT_MODE", "always")
    if draft_mode not in DRAFT_MODES:
        log.error(
            f"INPUT_DRAFT_MODE={draft_mode!r} is invalid. "
            f"Must be one of {DRAFT_MODES}."
        )
        sys.exit(2)

    rate_limit_raw = _optional_env("INPUT_RATE_LIMIT_DAYS", "7")
    try:
        rate_limit_days = int(rate_limit_raw)
    except ValueError:
        log.error(
            f"INPUT_RATE_LIMIT_DAYS={rate_limit_raw!r} is not an integer."
        )
        sys.exit(2)

    guardrails_raw = _optional_env("INPUT_GUARDRAILS_ALLOWLIST", "")
    guardrails_allowlist = (
        [p.strip() for p in guardrails_raw.split(",") if p.strip()]
        if guardrails_raw
        else []
    )

    return Target(
        repo=repo,
        interest_id=interest_id,
        min_confidence=_optional_env("INPUT_MIN_CONFIDENCE", "moderate"),
        rate_limit_days=rate_limit_days,
        draft_mode=draft_mode,
        guardrails_allowlist=guardrails_allowlist,
        notes="",
    )


def main():
    target = build_target_from_env()
    log.info(f"=== {target.repo} ===")
    log.info(f"  interest_id={target.interest_id}")
    log.info(f"  min_confidence={target.min_confidence}  "
             f"draft_mode={target.draft_mode}  "
             f"rate_limit_days={target.rate_limit_days}")

    try:
        result = process_target(target)
    except Exception as e:
        log.exception(f"  ✗ unhandled error: {e}")
        result = {"repo": target.repo, "status": "error", "error": str(e)}

    print("\n=== RUN SUMMARY ===")
    print(json.dumps(result, indent=2))

    # Surface key outputs to the GitHub Actions runner so consuming
    # workflows can branch on the result (e.g., notify Slack on
    # pr_opened, alert on rejected_path_violations).
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        try:
            with open(github_output, "a") as f:
                f.write(f"status={result.get('status', 'unknown')}\n")
                if "pr_url" in result:
                    f.write(f"pr_url={result['pr_url']}\n")
                if "issue_url" in result:
                    f.write(f"issue_url={result['issue_url']}\n")
                if "arxiv" in result:
                    f.write(f"arxiv={result['arxiv']}\n")
                if "tier" in result:
                    f.write(f"tier={result['tier']}\n")
        except OSError as e:
            log.warning(f"Could not write to $GITHUB_OUTPUT: {e}")

    # Non-zero exit on error so the workflow step fails visibly.
    if result.get("status") == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
