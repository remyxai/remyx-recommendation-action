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

import ast
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
REMYX_RECOMMENDATION_LIMIT = int(os.environ.get("REMYX_RECOMMENDATION_LIMIT", "25"))
# Max seconds to wait for recommendations to populate after triggering a
# refresh on an interest whose pool is empty (e.g. a brand-new interest
# whose daily ranking hasn't run yet). Polled, not a hard sleep.
REMYX_REFRESH_WAIT_S = int(os.environ.get("REMYX_REFRESH_WAIT_S", "150"))

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
#
# Permissive on the target package because §2 of the "ready-to-ship PRs"
# work requires the agent to be able to add small wiring edits to
# existing files (e.g. a 3-line hook in evaluation.py). The post-hoc
# check_integration() validator caps how much can change per existing
# file and rejects runs that only add freestanding modules.
# Python source anywhere in the repo is editable: a wiring edit has to be
# able to reach the real call site, which often lives outside the target
# package (a pipeline/stage driver, an entrypoint module, etc.), and we
# don't want to hard-code any one repo's directory layout. Infra files that
# happen to sit alongside source — container builds, shell scripts,
# dependency/build manifests, CI config — are blocked by ROLE in
# ALWAYS_BLOCKED, which takes precedence. The 50-line edit cap and the
# invocation check in check_integration() keep edits surgical and honest.
DEFAULT_ALLOWLIST_GLOBS = [
    "*.py",
    ".remyx-recommendation/**",
    "README.md",
]

# Cap on additions+deletions per pre-existing file. Keeps wiring edits
# small and surgical; rejects runs where Claude rewrote an unrelated
# module under the cover of "integration".
MAX_LINES_PER_EXISTING_FILE = 50

# Cap on number of newly-created .py files in the target package. A
# real integration adds one module, sometimes two; anything beyond
# that is scaffold-shaped.
MAX_NEW_PACKAGE_FILES = 3

# Stub density (fraction of function bodies that are pass / ellipsis /
# raise NotImplementedError / docstring-only) above which we route to
# Issue instead of opening a PR. At this density the paper's actual
# contribution isn't really present in the diff.
STUB_DENSITY_DOWNGRADE_THRESHOLD = 0.5

BUNDLE_DIR_NAME = ".remyx-recommendation"
BRANCH_PREFIX = "remyx-recommendation/"
PR_TITLE_PREFIX = "[Remyx Recommendation]"

# Files that are NEVER allowed to be touched. Blocked by ROLE (filename /
# type), not by directory, so the policy doesn't encode any one repo's
# layout: a Dockerfile is off-limits whether it sits at the root, under
# docker/, or anywhere else. `*` crosses `/` in path_matches_glob, so each
# pattern catches the file at the repo root and nested at any depth. This
# is checked before the allowlist and takes precedence, so even though
# `*.py` is allowlisted, build scripts and dependency manifests stay
# protected. (Replaces the old directory-based `docker/**` / `pipelines/**`
# / `config/**` blanket blocks, which were overfit to one repo's tree and
# locked out the stage drivers that are often the real call site.)
ALWAYS_BLOCKED = [
    ".github/**",            # CI / workflow config (GitHub-standard location)
    "*Dockerfile",           # container build recipes, anywhere
    "*Dockerfile.*",
    "*.dockerfile",
    "*.sh",                  # shell scripts (entrypoints, build hooks), anywhere
    "*requirements*.txt",    # pip dependency manifests, anywhere
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
    "MANIFEST.in",
    "*.lock",                # lockfiles (poetry.lock, uv.lock, …)
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

## How this maps onto your repo (candidate selection)

{selection_block}

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

SPEC.md names a PROPOSED CALL SITE under "How this maps onto your repo"
(the file + function the selection pass judged most implementable). Start
there, and keep exploration minimal — broad repo-wandering is the main
cost to avoid:
  - Open ONLY that file plus the modules its target function directly
    imports or calls. Read narrow line ranges, not whole files.
  - Use grep / symbol search to confirm the call site and local
    conventions. Do NOT list or read the whole `{package}/` or `tests/`
    tree.
  - Skip generated, vendored, lockfile, data, and notebook files, and any
    file over ~1500 lines unless the call site is inside it.
  - Once you can name the exact function you will call, STOP exploring and
    implement. Confirming the call site should take only a few reads.
Depth at the chosen call site is fine; breadth across the repo is not.

# Step 1 — decide: PR or Issue

DEFAULT: open an Issue. PR is the exception, not the rule.

Open as PR only if BOTH of these hold:

  (a) You can identify a SPECIFIC existing module/function in `{package}/`
      where this paper's contribution slots in (the "call site").

  (b) You can deliver the paper's CORE INSIGHT or RESULT as a useful,
      scoped change at that call site. You do NOT need to reproduce the
      paper's full method, architecture, training procedure, or reported
      numbers. A small change that moves the repo in the paper's
      direction (a scorer, filter, metric, evaluation hook, or focused
      behavior change) is the INTENDED deliverable, not a fallback.

Open as Issue if any of these is true:

  - The paper's contribution requires infrastructure the codebase lacks
    (a trainer when the repo is inference-only, a dataset format the
    repo never touches, external checkpoints with no path to load).
  - You cannot point at a specific existing call site to modify.
  - Your implementation would be a freestanding module that no existing
    code imports or calls — i.e. it could be deleted without breaking
    or altering anything in the repo.
  - You'd be inventing the integration rather than slotting into one.

If ANY of those hold, DO NOT WRITE CODE. Write a file at
`{issue_fallback_filename}` with this exact shape (Markdown):

```
# Title: short, action-oriented (becomes the Issue title)
Optional one-line subtitle.

## Why this paper is interesting for the team

(2-3 sentences from the spec + your own reading)

## What blocks a clean implementation

(Specifics: missing infra, no clear call site, required external
artifacts. Be concrete about what would need to exist for a real
integration to be drafted.)

## What we'd need to know / decide first

(1-3 questions or decisions the team should resolve before this becomes
implementable.)
```

The orchestrator detects this file and opens an Issue instead of a PR.
This is the HONEST outcome when the paper doesn't fit, not a failure.

# Step 2 — only if you're proceeding with PR: implement an INTEGRATION

The goal is the smallest change that calls into existing code and
delivers the paper's core insight as value to THIS repo. Implement the
RESULT, not the technique — do not port a trainer, model, or loss the repo
cannot host. NOT a scaffold. NOT a freestanding module.

Required outputs:

1. **At least one EDIT to an existing file** in `{package}/` that
   actually invokes your new code (the call site). A 3-line hook in
   `evaluation.py` that calls a new scorer is the model. Without this
   edit, the orchestrator will reject the run as scaffold-shaped.

   Keep each existing-file edit small — under ~50 lines net change.
   Larger edits get rejected.

2. **A capability-named module**, NOT `<paper-slug>_integration.py`.
   Pick a name that fits the repo's existing conventions and describes
   what the module DOES, not which paper it came from. Examples:
   `cot_grounding_check.py`, `pointcloud_quality.py`, `mask_refiner.py`.
   Paper attribution goes in the module docstring and README — never
   the filename. Keep the new file focused; if you need more than ~250
   lines, you're probably scaffolding.

3. **At least one new test that imports from a NON-NEW module** in
   `{package}/`. Pure self-tests of the new file don't prove
   integration. Example: a test that imports the existing call-site
   module, exercises the wiring edit you made, and asserts the
   integrated behavior.

4. **README append**: a short "(Capability) — adapted from (Paper Title)"
   section at the end. Attribute with this exact line (one Markdown
   link to the customer-facing Remyx product page):

       Contributed via [Remyx Recommendation]({attribution_url}).

   Do NOT use a different URL. The orchestrator's source repo is
   private; this link is the only one that resolves for external readers.

# Honesty rules

- If the public surface of your new module is dominated by `TODO`,
  `pass`, or `raise NotImplementedError` (more than ~half the
  function bodies), you are scaffolding. STOP and write the Issue file
  instead — the orchestrator will reject the run anyway.
- If your new module would import cleanly but never be called by
  anything else in the repo, STOP and write the Issue file instead.
- The Issue-mode path is the correct route when the paper doesn't fit.
  It is NOT a failure mode.

Run pytest before declaring done. If tests fail, fix them or scope down
to a smaller integration; do not modify files outside the guardrails
allowlist.

# CRITICAL: do not run git commands

You MUST NOT run any `git` command during your session (`git init`,
`git checkout`, `git stash`, `git reset`, `git commit`, `git add`,
`git rm`, `git rebase`, etc. — none of them, including `git status`).
The orchestrator manages all version control. Past runs have hit
subtle issues where agents ran a `git checkout` to back out a
half-edit and left the working tree in an orphan state that broke
the PR.

If you need to back out an edit, use the file-edit tools to restore
the file's content. Look up the original content via standard read
tools — do not invoke git.

When complete, output a one-paragraph SUMMARY of what you built. Call out:
  - Which existing file you modified (the call site)
  - Which new module you created (the capability name)
  - The paper insight this delivers, and what you intentionally scoped
    out as unnecessary for that value — frame these as scoping decisions,
    not shortfalls. A focused slice that delivers the result is success.

Still distinguish "intentionally out of scope" (expected) from
"stubbed / incomplete" (TODO-dominated bodies) — the latter still routes
to an Issue per the honesty rules above.
"""

# Two helper Claude prompts: PR/Issue routing pre-flight (§6) and the
# post-implementation self-review (§4). Both are rendered with str.replace()
# rather than str.format() so the literal `{` / `}` in JSON examples don't
# need to be doubled.

_PREFLIGHT_PROMPT_TEMPLATE = """\
You are routing a paper recommendation for the Remyx Recommendation
orchestrator. Decide: should the implementation step run (PR), or
should we open an Issue for the team to discuss first?

Inputs follow at the end of this message:
  1. The paper spec (title, abstract, why-this-paper, suggested experiment)
  2. A candidate-selection rationale (in the spec, under "How this maps
     onto your repo") — when present, a prior pass already judged this
     paper implementable against THIS repo and named the call sites and
     the implementable SUBSET it targets.
  3. The target repo's module layout

Evaluate the SCOPED implementation the selection rationale describes — the
implementable subset wired into the named call sites — NOT the paper's
full or maximal contribution. A paper whose maximal form needs missing
infra (a trainer, a renderer, a synthesis engine) can still be a sound PR
if the selection rationale identifies a real, smaller slice that drops
into an existing call site (e.g. consuming a paper's released benchmark
through the existing eval path, rather than rebuilding its data-generation
engine). Don't route to ISSUE merely because the paper's headline method
is heavy — judge the scoped slice.

Route to ISSUE only if any of these is likely true of THAT scoped slice:

  - Even the scoped implementation requires infrastructure that isn't in
    the repo (a trainer when the repo is inference-only, a data format the
    repo never touches, checkpoints with no loader path).
  - There is no clear call site — no existing module that naturally hosts
    even the scoped contribution (and the selection rationale, if present,
    names none that hold up against the layout).
  - The most realistic implementation would be a freestanding module that
    no existing code would call.

Otherwise route to PR.

Output a single JSON object. Start with `{` and end with `}`. No
Markdown fences, no prose before or after. Schema:

{
  "decision": "PR" | "ISSUE",
  "reasoning": "<2-3 sentences explaining the call>",
  "issue_title": "<if ISSUE: short, action-oriented title; else empty>",
  "issue_body": "<if ISSUE: Markdown body with sections 'Why this paper
                  is interesting for the team', 'What blocks a clean
                  implementation', 'What we'd need to know / decide
                  first'; else empty>"
}

--- Paper spec ---

__SPEC__

--- Repo layout (top-level modules in the target package + tests) ---

__LAYOUT__
"""

_SELECTION_PROMPT_TEMPLATE = """\
You are selecting which paper recommendation the Remyx Recommendation
orchestrator should implement as a draft PR against the target repo.

You are given a ranked list of candidate papers (ranked by Remyx
relevance, highest first) and the target repo's module layout. Relevance
rank is NOT implementability: the top-ranked paper is frequently a model
architecture or a training method with no call site in a data / inference
pipeline, while a lower-ranked candidate is a clean drop-in.

Pick the ONE candidate that is most directly implementable as a focused
PR against THIS repo. Prefer a candidate that:
  - maps onto an existing module / call site visible in the layout,
  - is a pipeline / data-generation / eval change the repo can actually
    host (not a new trainer, model architecture, or checkpoint the repo
    has no loader for),
  - ships its contribution as code this repo would call, rather than a
    freestanding module nothing imports.

Down-rank candidates whose primary contribution is a model to be trained,
an architecture, or anything needing infrastructure absent from the
layout — even when they rank higher by relevance.

Output a single JSON object. Start with `{` and end with `}`. No Markdown
fences, no prose before or after. Schema:

{
  "chosen_index": <integer index into the candidate list below>,
  "reasoning": "<2-3 sentences: why this candidate is the most directly
                 implementable against this repo, naming the call site>",
  "rejected": [
    {"index": <int>, "why": "<one line: why this candidate is a worse fit
                              to implement now, e.g. needs a trainer the
                              repo lacks>"}
  ]
}

--- Candidates (highest relevance first) ---

__CANDIDATES__

--- Repo layout (top-level modules in the target package + tests) ---

__LAYOUT__
"""

_SELF_REVIEW_PROMPT_TEMPLATE = """\
You are reviewing your own implementation of a paper recommendation
before the orchestrator opens a PR.

Inputs:
  1. The original implementation spec (read `.remyx-recommendation/SPEC.md`
     in the working directory)
  2. The full diff of your changes (provided at the end of this message)

Output a single JSON object. Start with `{` and end with `}`. No
Markdown fences, no prose before or after. Schema:

{
  "implemented": [<bullets describing what from the paper's method is
                   concretely implemented in your diff>],
  "stubbed":     [<bullets describing what from the paper is left out,
                   with required infra noted in parentheses>],
  "call_site":   "<which existing entry point the new code is invoked
                   from, or '(none)' if nothing in the product calls it>",
  "is_orphan":   <true if the new code is NOT reached from any pre-existing
                   execution path — no production / pipeline entry point
                   and no existing module invokes it (only the tests you
                   added, if any, call it). This is about REACHABILITY, not
                   quality: rich, correct code that the product never calls
                   is still an orphan. Do NOT use this field to judge
                   whether the code is "too simple" — triviality is scored
                   separately by stub density.>,
  "honest_summary": "<one short paragraph: what you actually built,
                     what's missing, whether the paper's primary
                     contribution is really present>"
}

Be ruthless about reachability. If the only thing that calls your new code
is a test you added (or nothing at all), set is_orphan=true — the product
never exercises it. If a pre-existing entry point (a pipeline/stage driver,
a CLI, an existing module) now invokes your new code, set is_orphan=false.
Separately, if you only implemented the plumbing around the paper's
contribution but not the contribution itself, list those parts as stubbed.

--- Diff ---

__DIFF__
"""

_PR_BODY_TEMPLATE = """\
> **Drafted by an autonomous discovery loop** — Remyx ranks recent arXiv papers against this team's research interest and shipping history; Claude Code selects the candidate most directly implementable against this repo from the lookback window and drafts it.
>
> **Recommended paper**: [{paper_title}](https://arxiv.org/abs/{arxiv_id})
> **Confidence**: {tier_emoji} {tier} (Remyx relevance {relevance_score:.2f})
> **Research interest**: {interest_name}
> **Implementation by**: Claude Code as autonomous agent

---

## Why this paper for this team

{reasoning}
{selection_section}
## Suggested experiment

{suggested_experiment}

---

{test_section}

---

_Opened by the [Remyx Recommendation]({attribution_url}) orchestrator._
"""


# ─── Data classes ──────────────────────────────────────────────────────────


DRAFT_MODES = ("always", "on_test_failure", "never")

# Terminal statuses that should make the workflow step exit non-zero (red
# in CI). Everything else — Issues, skips, PRs — is a legitimate green
# outcome. `claude_failed` used to exit 0, so a run that produced no PR/Issue
# looked green; it now fails visibly.
FAILURE_EXIT_STATUSES = {"error", "claude_failed"}


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
    # Per-run wall-clock budget for the Claude Code implementation step.
    # 600s was too tight on large repos; configurable via `claude-timeout`.
    claude_timeout_s: int = 900
    # Optional: force-select a specific candidate by arxiv_id (skips the
    # LLM selection pass) so eval re-runs are reproducible. Empty = normal
    # selection.
    pin_arxiv: str = ""
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


def _remyx_post(path: str, body: dict) -> dict:
    """POST against the Remyx engine API with the configured API key.
    Raises RuntimeError on non-2xx response. Mirrors ``_remyx_get``."""
    api_key = os.environ.get("REMYX_API_KEY") or os.environ.get("REMYXAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "REMYX_API_KEY (or REMYXAI_API_KEY) is required. Generate one "
            "from your engine.remyx.ai settings and add it as a workflow "
            "secret."
        )
    url = REMYX_API_BASE.rstrip("/") + path
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "feature-finder-orchestrator",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")[:400]
        raise RuntimeError(
            f"Remyx API POST {path} → HTTP {e.code}: {body_text}"
        ) from e


def _refresh_and_poll_recommendations(target: Target, fetch_fn) -> list:
    """Trigger a recommendation refresh for the interest, then poll until
    picks appear or ``REMYX_REFRESH_WAIT_S`` elapses.

    A brand-new interest (or one whose daily ranking hasn't run since the
    last cron) returns an empty pool; the engine ranks asynchronously after
    a POST to /papers/recommended/refresh. Returns the populated list, or
    [] if nothing landed within the budget.
    """
    log.info("  → empty recommendation pool; triggering "
             "/papers/recommended/refresh and polling")
    try:
        _remyx_post(
            "/api/v1.0/papers/recommended/refresh",
            {"interest_id": target.interest_id},
        )
    except Exception as e:
        log.warning(f"    (refresh trigger failed: {e})")
    deadline = time.monotonic() + REMYX_REFRESH_WAIT_S
    while time.monotonic() < deadline:
        time.sleep(10)
        try:
            papers = fetch_fn()
        except Exception as e:
            log.warning(f"    (poll failed: {e}; retrying)")
            continue
        if papers:
            log.info(f"    ✓ recommendations populated ({len(papers)})")
            return papers
    return []


def _relevance_to_tier(score: float) -> str:
    if score >= RELEVANCE_TIER_FLOOR["high"]:
        return "high"
    if score >= RELEVANCE_TIER_FLOOR["moderate"]:
        return "moderate"
    if score >= RELEVANCE_TIER_FLOOR["low"]:
        return "low"
    return "noise"


def _fetch_interest_context(interest_id: str) -> tuple[str, str]:
    """Fetch the interest's name + rich-text focus body once per run.

    Returns (interest_name, interest_context). The context body is the
    rich text the customer wrote on engine.remyx.ai about their research
    focus / goals — it gives Claude Code a far better picture of what to
    build than the paper abstract + reasoning alone. Best-effort: on any
    failure we return empty strings and fall back to the reasoning-only
    brief.
    """
    try:
        interest = _remyx_get(f"/api/v1.0/research-interests/{interest_id}")
        return (
            (interest.get("name") or ""),
            (interest.get("context") or "").strip(),
        )
    except Exception as e:
        log.warning(f"    (interest context fetch failed: {e}; "
                    f"continuing with reasoning-only brief)")
        return "", ""


def _paper_to_recommendation(
    paper: dict, fallback_interest_name: str, interest_context: str
) -> Recommendation:
    """Map one /papers/recommended envelope entry to a Recommendation."""
    relevance = float(paper.get("relevance_score") or 0.0)
    resource = paper.get("resource") or {}
    arxiv_id = paper.get("resource_id") or resource.get("arxiv_id") or ""
    abstract = (resource.get("abstract") or resource.get("summary") or "").strip()
    return Recommendation(
        paper_title=paper.get("title") or "(untitled)",
        arxiv_id=arxiv_id,
        tier=_relevance_to_tier(relevance),
        z_score=0.0,                       # legacy field, unused
        spec_md="",                        # legacy; rendered from fields below
        paper_abstract=abstract,
        team_context="",                   # Remyx keeps team context server-side
        domain_summary="",
        raw_paper_md="",
        relevance_score=relevance,
        reasoning=(paper.get("reasoning") or "").strip(),
        suggested_experiment=(paper.get("suggested_experiment") or "").strip(),
        recommendation_id=paper.get("recommendation_id") or "",
        interest_name=paper.get("interest_name") or fallback_interest_name,
        interest_context=interest_context,
    )


def query_remyx_candidates(target: Target) -> list[Recommendation]:
    """Pull the top-N recommendations for ``target.interest_id`` over the
    configured lookback window and return them as a relevance-ranked list.

    The window is ``REMYX_RECOMMENDATION_PERIOD`` (default ``"week"`` — the
    past 7 days) and the pool size is ``REMYX_RECOMMENDATION_LIMIT``
    (default 25), both surfaced as the ``lookback`` / ``candidate-pool``
    action inputs. Remyx owns commit-history extraction, candidate pool,
    embedding pre-filter, Gemini ranking, and reasoning generation; the
    action is a pure consumer.

    The earlier behaviour took only ``papers[0]``, which wasted the
    lookback: the top-ranked paper is often a model-architecture or
    training-method paper with no call site in a data-pipeline repo, while
    a lower-ranked candidate is a clean drop-in. Returning the full pool
    lets ``select_recommendation`` pick the most implementable candidate.

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
             f"(interest={target.interest_id[:8]}…, "
             f"period={REMYX_RECOMMENDATION_PERIOD}, "
             f"limit={REMYX_RECOMMENDATION_LIMIT})")
    def _fetch_papers() -> list:
        resp = _remyx_get(
            "/api/v1.0/papers/recommended",
            params={
                "interest_id": target.interest_id,
                "period":      REMYX_RECOMMENDATION_PERIOD,
                "limit":       REMYX_RECOMMENDATION_LIMIT,
            },
        )
        return resp.get("papers") or []

    papers = _fetch_papers()
    if not papers:
        # A brand-new interest (or one whose daily refresh hasn't run since
        # the last cron) has no ranked picks yet. Trigger a refresh and poll
        # rather than failing the run outright.
        papers = _refresh_and_poll_recommendations(target, _fetch_papers)
    if not papers:
        raise RuntimeError(
            f"Remyx returned no recommendations for interest "
            f"{target.interest_id} in period={REMYX_RECOMMENDATION_PERIOD} "
            f"even after triggering /papers/recommended/refresh and waiting "
            f"{REMYX_REFRESH_WAIT_S}s. The interest may have no fresh picks "
            f"in this window."
        )

    interest_name, interest_context = _fetch_interest_context(target.interest_id)
    candidates = [
        _paper_to_recommendation(p, interest_name, interest_context)
        for p in papers
    ]
    for i, c in enumerate(candidates):
        log.info(f"    [{i}] {c.paper_title[:55]}…  "
                 f"relevance={c.relevance_score:.2f}  tier={c.tier}")
    return candidates


def query_remyx_recommendation(target: Target) -> Recommendation:
    """Back-compat shim: the single highest-ranked recommendation.

    Retained for callers / tests that only want the top pick. The
    orchestrator now calls ``query_remyx_candidates`` and runs a
    selection pass over the full pool instead.
    """
    return query_remyx_candidates(target)[0]



# ─── Dedup ─────────────────────────────────────────────────────────────────


def existing_pr_for(target: Target, branch: str) -> dict | None:
    """Return the PR dict if an open PR exists on the target repo for `branch`."""
    head_owner = target.repo.split("/")[0]
    head = f"{head_owner}:{branch}"
    prs = gh_api("GET", f"/repos/{target.repo}/pulls?state=open&head={head}")
    return prs[0] if prs else None


def open_remyx_issues(target: Target) -> list[dict]:
    """Open Remyx Recommendation Issues on the target repo.

    GitHub's /issues endpoint also returns PRs (they carry a
    'pull_request' key) — those are filtered out; PRs are deduped
    separately by existing_pr_for. We keep only items that look like one
    of ours: the title carries the PR_TITLE_PREFIX or the body has the
    orchestrator's attribution footer. Bounded to the first 100 open
    issues (same pragmatic cap as recent_pr_within_rate_limit)."""
    issues = gh_api(
        "GET", f"/repos/{target.repo}/issues?state=open&per_page=100"
    ) or []
    ours = []
    for it in issues:
        if it.get("pull_request"):
            continue
        title = it.get("title") or ""
        body = it.get("body") or ""
        if title.startswith(PR_TITLE_PREFIX) or "Remyx Recommendation" in body:
            ours.append(it)
    return ours


def issue_for_paper(open_issues: list[dict], rec: Recommendation) -> dict | None:
    """Return an already-open Remyx Issue for this paper, if any.

    Matched on the arxiv_id in the issue body — every Remyx issue links
    ``arxiv.org/abs/<id>``, and that survives the OPEN_AS_ISSUE path where
    the title is Claude-authored rather than ``<prefix> <paper_title>``.
    Falls back to an exact title match when the recommendation carries no
    arxiv_id. Pure (no network) so the matching is unit-testable; the
    fetch lives in open_remyx_issues."""
    needle = f"arxiv.org/abs/{rec.arxiv_id}" if rec.arxiv_id else None
    title_match = f"{PR_TITLE_PREFIX} {rec.paper_title}"
    for it in open_issues:
        if needle and needle in (it.get("body") or ""):
            return it
        if not rec.arxiv_id and (it.get("title") or "") == title_match:
            return it
    return None


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
    # Skip Git-LFS smudge: the orchestrator only reads code structure and
    # makes small edits — it never needs the LFS blobs (model weights,
    # datasets). Fetching them is slow, and a repo whose LFS bandwidth
    # budget is exhausted fails the clone outright ("exceeded its LFS
    # budget") even though every file we touch is plain text. Pointer files
    # are checked out instead.
    clone_env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    subprocess.run(
        ["git", "clone", "--depth", "20", repo_url, str(workdir)],
        check=True, env=clone_env,
    )
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


def write_spec_bundle(
    workdir: Path, target: Target, rec: Recommendation, package: str,
    selection_note: str = "",
) -> None:
    """Write the .remyx-recommendation/ bundle that Claude Code reads as its brief.

    ``selection_note`` is the candidate-selection rationale: why this
    paper was picked from the pool as the most implementable against THIS
    repo, including the call sites it targets. It's written into the spec
    so BOTH the pre-flight routing pass and the implementer evaluate the
    same scoped framing the selection pass reasoned about — without it,
    pre-flight re-derives PR-vs-Issue from the abstract alone and can
    contradict the selection (e.g. judging a benchmark paper's maximal
    form needs infra the repo lacks, while the selection identified an
    implementable subset).
    """
    bundle = workdir / BUNDLE_DIR_NAME
    bundle.mkdir(exist_ok=True)

    interest_block = (
        rec.interest_context
        if rec.interest_context
        else "(no research-focus body configured for this interest on engine.remyx.ai)"
    )
    note = (selection_note or "").strip()
    selection_block = (
        note
        if note and not note.startswith("(")
        else "(no separate selection rationale — this was the top-ranked candidate)"
    )
    (bundle / "SPEC.md").write_text(_SPEC_MD_TEMPLATE.format(
        paper_title=rec.paper_title,
        arxiv_id=rec.arxiv_id,
        tier=rec.tier,
        relevance_score=rec.relevance_score,
        interest_name=rec.interest_name or "(unnamed interest)",
        interest_context_block=interest_block,
        reasoning=rec.reasoning or "(no reasoning provided)",
        selection_block=selection_block,
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

    allowlist = effective_allowlist(target, package)
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


def invoke_claude_code(workdir: Path, timeout_s: int = 900) -> tuple[bool, str]:
    """Invoke the Claude Code CLI in headless mode with the workdir as context.

    Returns (success, stdout/stderr). Success means CLI exit 0 — caller still
    validates the produced changes with the path-allowlist check + tests.

    ``REMYX_CLAUDE_MAX_TURNS`` (optional) caps the agent's tool-use turns to
    bound cost; unset means no cap (avoids truncating legitimate work).
    """
    invocation = (workdir / BUNDLE_DIR_NAME / "INVOCATION.md").read_text()
    log.info(f"  → invoking Claude Code (timeout={timeout_s}s) in {workdir}")
    cmd = ["claude", "--dangerously-skip-permissions"]
    max_turns = os.environ.get("REMYX_CLAUDE_MAX_TURNS", "").strip()
    if max_turns:
        cmd += ["--max-turns", max_turns]
    cmd += ["-p", invocation]
    try:
        # `claude` CLI: -p for prompt, --dangerously-skip-permissions for CI
        result = subprocess.run(
            cmd,
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


# ─── Pre-flight routing + self-review (§4, §6) ─────────────────────────────


def _run_claude_oneshot(
    workdir: Path, prompt: str, timeout_s: int
) -> tuple[bool, str]:
    """Run the Claude CLI headless with `prompt` and return (ok, stdout).

    Used for the pre-flight routing and the self-review passes — both
    expect a JSON object back, not a full code-generation session.
    Failures here are non-fatal: the orchestrator falls through to the
    normal implementation flow.
    """
    try:
        result = subprocess.run(
            ["claude", "--dangerously-skip-permissions", "-p", prompt],
            cwd=workdir, capture_output=True, text=True, timeout=timeout_s,
        )
        return result.returncode == 0, (result.stdout or "")
    except subprocess.TimeoutExpired:
        return False, f"claude CLI timed out after {timeout_s}s"
    except FileNotFoundError:
        return False, "claude CLI not found on PATH"


def _extract_json_object(s: str) -> dict | None:
    """Pull the first JSON object out of `s`. Tolerant of prose wrappers."""
    if not s:
        return None
    try:
        start = s.index("{")
        end = s.rindex("}")
    except ValueError:
        return None
    try:
        return json.loads(s[start:end + 1])
    except json.JSONDecodeError:
        return None


def _repo_layout_manifest(workdir: Path, package: str, max_lines: int = 60) -> str:
    """Short module-by-module manifest of the target repo for pre-flight.

    Lists the .py files under `{package}/` with the first line of their
    module docstring (where present) and the names of the test files
    under `tests/`. Capped to `max_lines` to keep the prompt cheap.
    """
    lines: list[str] = []
    pkg_dir = workdir / package
    if pkg_dir.is_dir():
        py_files = sorted(pkg_dir.rglob("*.py"))
        lines.append(f"# {package}/ ({len(py_files)} modules)")
        for p in py_files:
            rel = p.relative_to(workdir).as_posix()
            doc_first = ""
            try:
                doc = ast.get_docstring(ast.parse(p.read_text())) or ""
                doc_first = doc.splitlines()[0] if doc else ""
            except (SyntaxError, OSError):
                pass
            if doc_first:
                lines.append(f"  {rel}  — {doc_first[:80]}")
            else:
                lines.append(f"  {rel}")
    tests_dir = workdir / "tests"
    if tests_dir.is_dir():
        test_files = sorted(tests_dir.rglob("test_*.py"))[:20]
        if test_files:
            lines.append(f"\n# tests/ ({len(test_files)} files shown)")
            for p in test_files:
                lines.append(f"  {p.relative_to(workdir).as_posix()}")
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"  ... ({len(lines) - max_lines} more)"]
    return "\n".join(lines) or "(empty)"


def preflight_routing(
    workdir: Path, package: str, timeout_s: int = 180
) -> dict | None:
    """Cheap Claude pass that decides PR vs Issue BEFORE implementation.

    Returns the parsed JSON ({decision, reasoning, issue_title, issue_body})
    or None on any failure (parse error, timeout, missing CLI). On None
    the orchestrator falls through to the regular implementation flow,
    so a failed pre-flight never blocks a PR — it just doesn't save the
    Claude budget.
    """
    spec_path = workdir / BUNDLE_DIR_NAME / "SPEC.md"
    if not spec_path.exists():
        return None
    spec_md = spec_path.read_text()
    layout = _repo_layout_manifest(workdir, package)
    prompt = (
        _PREFLIGHT_PROMPT_TEMPLATE
        .replace("__SPEC__", spec_md)
        .replace("__LAYOUT__", layout)
    )
    log.info("  → pre-flight routing pass (PR vs Issue)")
    ok, output = _run_claude_oneshot(workdir, prompt, timeout_s)
    if not ok:
        log.warning(f"  pre-flight call failed: {output[:200]}; "
                    f"falling through to implementation")
        return None
    data = _extract_json_object(output)
    if data is None:
        log.warning(f"  pre-flight: couldn't parse JSON; raw: {output[:300]!r}")
        return None
    decision = str(data.get("decision") or "").upper()
    if decision not in ("PR", "ISSUE"):
        log.warning(f"  pre-flight: invalid decision {decision!r}; "
                    f"falling through to implementation")
        return None
    data["decision"] = decision
    log.info(f"  pre-flight decision: {decision} — "
             f"{(data.get('reasoning') or '')[:120]}")
    return data


def _render_candidate_brief(candidates: list[Recommendation]) -> str:
    """Numbered, relevance-ranked brief of the candidate pool for the
    selection pass. Index matches list position so the model's
    ``chosen_index`` maps straight back."""
    blocks: list[str] = []
    for i, c in enumerate(candidates):
        abstract = " ".join((c.paper_abstract or "").split())
        blocks.append(
            f"[{i}] {c.paper_title}  "
            f"(arxiv {c.arxiv_id or 'n/a'}, relevance {c.relevance_score:.2f}, "
            f"tier {c.tier})\n"
            f"    why surfaced: {(c.reasoning or '(none)')[:600]}\n"
            f"    abstract: {abstract[:400]}"
        )
    return "\n\n".join(blocks)


def select_recommendation(
    workdir: Path, package: str, candidates: list[Recommendation],
    timeout_s: int = 180,
) -> dict | None:
    """Claude pass that picks the most implementable candidate from the
    lookback pool, given the target repo's module layout.

    Returns the parsed JSON ({chosen_index, reasoning, rejected}) or None
    on any failure (single candidate, parse error, out-of-range index,
    timeout, missing CLI). On None the caller falls back to candidates[0]
    (the highest-ranked), preserving the pre-selection behaviour.

    This only chooses *which* candidate to implement — it never decides
    PR vs Issue. The chosen candidate still runs the full preflight +
    integration / stub / test / self-review gate chain, any of which can
    downgrade to an Issue.
    """
    if len(candidates) <= 1:
        return None
    layout = _repo_layout_manifest(workdir, package)
    prompt = (
        _SELECTION_PROMPT_TEMPLATE
        .replace("__CANDIDATES__", _render_candidate_brief(candidates))
        .replace("__LAYOUT__", layout)
    )
    log.info(f"  → selection pass over {len(candidates)} candidates")
    ok, output = _run_claude_oneshot(workdir, prompt, timeout_s)
    if not ok:
        log.warning(f"  selection call failed: {output[:200]}; "
                    f"falling back to top-ranked candidate")
        return None
    data = _extract_json_object(output)
    if data is None:
        log.warning(f"  selection: couldn't parse JSON; raw: {output[:300]!r}")
        return None
    try:
        idx = int(data.get("chosen_index"))
    except (TypeError, ValueError):
        log.warning(f"  selection: chosen_index not an int "
                    f"({data.get('chosen_index')!r}); falling back")
        return None
    if not (0 <= idx < len(candidates)):
        log.warning(f"  selection: chosen_index {idx} out of range "
                    f"[0,{len(candidates)}); falling back")
        return None
    data["chosen_index"] = idx
    log.info(f"  selection: candidate [{idx}] "
             f"{candidates[idx].paper_title[:50]}… — "
             f"{(data.get('reasoning') or '')[:120]}")
    return data


def self_review_diff(
    workdir: Path, timeout_s: int = 180
) -> dict | None:
    """Second Claude pass over the diff. Returns the parsed JSON or None.

    Never raises and never blocks: a failure here just means the PR
    won't get the self-review section. The integration / stub-density
    checks are the load-bearing gates.
    """
    try:
        diff_proc = subprocess.run(
            ["git", "diff", "HEAD"],
            cwd=workdir, capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as e:
        log.warning(f"  self-review: git diff failed ({e}); skipping")
        return None
    diff = diff_proc.stdout
    if not diff.strip():
        return None
    # Cap diff size at ~80KB to keep the prompt cheap and well under
    # any context limit the headless CLI imposes.
    if len(diff) > 80_000:
        diff = diff[:80_000] + "\n... (truncated)"
    prompt = _SELF_REVIEW_PROMPT_TEMPLATE.replace("__DIFF__", diff)
    log.info(f"  → self-review pass (diff={len(diff)} bytes)")
    ok, output = _run_claude_oneshot(workdir, prompt, timeout_s)
    if not ok:
        log.warning(f"  self-review call failed: {output[:200]}")
        return None
    data = _extract_json_object(output)
    if data is None:
        log.warning(f"  self-review: couldn't parse JSON; raw: {output[:300]!r}")
        return None
    return data


def _render_self_review_section(review: dict) -> str:
    """Render the self-review JSON into a PR-body section prepended above
    the test results. Always returns a complete Markdown block ending
    in a blank line."""
    impl = review.get("implemented") or []
    stubbed = review.get("stubbed") or []
    call_site = review.get("call_site") or "(unspecified)"
    summary = (review.get("honest_summary") or "").strip()

    def _bullets(items: list) -> str:
        if not items:
            return "_(none reported)_"
        return "\n".join(f"- {x}" for x in items)

    parts = [
        "## What this PR actually does",
        "",
        f"**Call site**: `{call_site}`",
        "",
        "**Implemented from the paper**:",
        _bullets(impl),
        "",
        "**Stubbed / left out**:",
        _bullets(stubbed),
    ]
    if summary:
        parts += ["", f"_{summary}_"]
    parts.append("")
    return "\n".join(parts)


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
    # --untracked-files=all lists individual files inside a newly-created
    # directory instead of collapsing them to "newdir/". Without it, a new
    # file in a brand-new dir (e.g. a first-ever tests/ folder) shows up as
    # the directory, which the path-allowlist and integration/invocation
    # checks can't reason about per-file.
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
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
    """Simple glob matcher. `**` matches any number of path segments,
    INCLUDING zero; `*` matches within a segment.

    fnmatch alone treats `*` as crossing `/`, so `tests/**/*.py` only
    matches when there's at least one intermediate dir — it rejects a
    top-level `tests/test_foo.py`, which is exactly the shape the §3 test
    gate expects. We test three normalizations per pattern so the
    zero-segment case matches too:
      - the raw pattern,
      - `**` → `*`            (collapse to single star),
      - `**/` → ``            (drop the segment entirely, so
                               `tests/**/*.py` also matches `tests/x.py`).
    """
    import fnmatch
    # Case-insensitive: `fnmatch.fnmatch` is case-sensitive on Linux, which
    # rejected e.g. a repo's `README.MD` against the `README.md` allowlist
    # entry and threw away an otherwise-valid PR.
    lower_path = path.lower()
    for p in patterns:
        variants = {p, p.replace("**", "*"), p.replace("**/", "")}
        if any(fnmatch.fnmatch(lower_path, v.lower()) for v in variants):
            return True
    return False


def effective_allowlist(target: Target, package: str) -> list[str]:
    """The default allowlist globs (with `{package}` filled in) PLUS any
    extra globs the customer passed via `guardrails-allowlist`.

    The customer input EXTENDS the defaults — it does not replace them. The
    old `target.guardrails_allowlist or [defaults]` short-circuit silently
    dropped the defaults (`.remyx-recommendation/**`, `*.py`, `README.md`)
    the moment any extra glob was supplied, which then flagged the agent's
    own scaffolding files as violations.
    """
    base = [g.format(package=package) for g in DEFAULT_ALLOWLIST_GLOBS]
    extra = [g for g in (target.guardrails_allowlist or []) if g not in base]
    return base + extra


def validate_changes(workdir: Path, target: Target, package: str) -> tuple[bool, list[str]]:
    """Returns (passed_allowlist, violations)."""
    allowlist = effective_allowlist(target, package)
    paths = changed_files(workdir)
    violations = []
    for p in paths:
        if path_matches_glob(p, ALWAYS_BLOCKED):
            violations.append(f"BLOCKED: {p}")
            continue
        if not path_matches_glob(p, allowlist):
            violations.append(f"NOT IN ALLOWLIST: {p}")
    return (not violations, violations)


# ─── Integration / stub-density / test-integration validators ──────────────
#
# These run AFTER the path-allowlist check passes. They enforce the
# "ready-to-ship PRs" shape: a small wiring edit to an existing file
# that calls into a new capability-named module, with at least one
# test that touches an existing module, and a non-stub-dominated new
# module. Failing any of these routes the run to Issue instead of PR.


def _file_is_new(workdir: Path, path: str) -> bool:
    """True if `path` did not exist at HEAD (i.e. Claude created it)."""
    result = subprocess.run(
        ["git", "ls-tree", "HEAD", "--", path],
        cwd=workdir, capture_output=True, text=True, check=False,
    )
    return not result.stdout.strip()


def _diff_line_changes(workdir: Path, path: str) -> tuple[int, int]:
    """Return (added, deleted) lines for `path` vs HEAD."""
    result = subprocess.run(
        ["git", "diff", "--numstat", "HEAD", "--", path],
        cwd=workdir, capture_output=True, text=True, check=False,
    )
    out = result.stdout.strip()
    if not out:
        return 0, 0
    parts = out.split("\t", 2)
    if len(parts) < 2:
        return 0, 0
    try:
        added = int(parts[0]) if parts[0] != "-" else 0
        deleted = int(parts[1]) if parts[1] != "-" else 0
    except ValueError:
        return 0, 0
    return added, deleted


def _head_source(workdir: Path, path: str) -> str:
    """Source of `path` at HEAD, or '' if it didn't exist there."""
    r = subprocess.run(
        ["git", "show", f"HEAD:{path}"],
        cwd=workdir, capture_output=True, text=True, check=False,
    )
    return r.stdout if r.returncode == 0 else ""


def _public_callables(src: str) -> set[str]:
    """Names of public functions, methods, and classes defined in `src`.

    Methods are included by their bare name because an invocation
    `obj.method(...)` is matched on the attribute name (see _called_names).
    Underscore-prefixed names are treated as private and ignored.
    """
    names: set[str] = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return names
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                names.add(node.name)
    return names


def _called_names(src: str) -> set[str]:
    """Names appearing in a call position in `src`: `foo(...)` yields
    'foo', `obj.foo(...)` yields 'foo'."""
    called: set[str] = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return called
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                called.add(fn.id)
            elif isinstance(fn, ast.Attribute):
                called.add(fn.attr)
    return called


def _added_callables(workdir: Path, path: str) -> set[str]:
    """Public callables defined in the working-tree `path` that were not
    defined at HEAD — the functions / methods / classes this diff adds."""
    if not path.endswith(".py"):
        return set()
    try:
        current = (workdir / path).read_text()
    except OSError:
        return set()
    now = _public_callables(current)
    if _file_is_new(workdir, path):
        return now
    return now - _public_callables(_head_source(workdir, path))


def check_integration(
    workdir: Path, target: Target, package: str
) -> tuple[bool, list[str]]:
    """Reject scaffold-shaped runs — code that's added but never called.

    Pass criteria — ALL of:
      * Number of new .py files under {package}/ ≤ MAX_NEW_PACKAGE_FILES.
      * Each modified existing file's net change ≤
        MAX_LINES_PER_EXISTING_FILE lines.
      * If the diff adds any new public function / method / class, at least
        one of them must be INVOKED from a different changed file. This
        proves the new code is wired into a call site rather than merely
        defined — and it covers both shapes: a brand-new module (called
        from a modified existing file) and methods/functions bolted onto an
        existing file (called from elsewhere in the diff). An import alone
        no longer counts; there must be an actual call.

    A newly-added symbol can only be reached by code also added/modified in
    this run (otherwise it would have been a NameError before), so scanning
    the changed set is sufficient. A test counts as a call site here — the
    code at least runs; whether a *production* path reaches it is the
    self-review reachability pass's job (§4).

    Returns (passed, [violations]).
    """
    paths = changed_files(workdir)
    pkg_prefix = f"{package}/"

    new_pkg_files = [
        p for p in paths
        if p.startswith(pkg_prefix) and p.endswith(".py") and _file_is_new(workdir, p)
    ]

    violations: list[str] = []

    if len(new_pkg_files) > MAX_NEW_PACKAGE_FILES:
        violations.append(
            f"too many new files in {package}/: {len(new_pkg_files)} > "
            f"{MAX_NEW_PACKAGE_FILES}"
        )

    for p in paths:
        if _file_is_new(workdir, p):
            continue
        added, deleted = _diff_line_changes(workdir, p)
        if added + deleted > MAX_LINES_PER_EXISTING_FILE:
            violations.append(
                f"oversized edit to existing file {p}: +{added}/-{deleted} "
                f"> {MAX_LINES_PER_EXISTING_FILE}"
            )

    # Invocation check. Every newly-added callable, keyed by the file that
    # defines it, must be called from some OTHER changed file.
    changed_py = [p for p in paths if p.endswith(".py")]
    added_by_file: dict[str, set[str]] = {}
    for p in changed_py:
        added = _added_callables(workdir, p)
        if added:
            added_by_file[p] = added

    if added_by_file:
        calls_by_file: dict[str, set[str]] = {}
        for p in changed_py:
            try:
                calls_by_file[p] = _called_names((workdir / p).read_text())
            except OSError:
                continue
        integrated: set[str] = set()
        for def_file, names in added_by_file.items():
            for call_file, calls in calls_by_file.items():
                if call_file == def_file:
                    continue
                integrated |= names & calls
        if not integrated:
            all_added = sorted({n for ns in added_by_file.values() for n in ns})
            shown = ", ".join(all_added[:8]) + ("…" if len(all_added) > 8 else "")
            violations.append(
                f"none of the newly-added functions/methods/classes are "
                f"invoked from another changed file — the diff defines code "
                f"nothing calls ({shown}). Wire the new capability into a "
                f"real call site (an existing module, a stage driver, or at "
                f"least a test that exercises it) or open as Issue."
            )

    return (not violations, violations)


def _is_stub_body(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Heuristic: is this function body just a placeholder?

    Treated as a stub:
      - body is a single `pass`
      - body is a single `...` (Ellipsis expression)
      - body is a single `raise NotImplementedError(...)`
      - body is docstring-only (no executable statements after it)

    Not treated as a stub:
      - return statements (even `return None`)
      - real expressions / calls
      - control flow
    """
    body = list(node.body)
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    if not body:
        return True
    if len(body) != 1:
        return False
    stmt = body[0]
    if isinstance(stmt, ast.Pass):
        return True
    if (isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and stmt.value.value is Ellipsis):
        return True
    if isinstance(stmt, ast.Raise) and stmt.exc is not None:
        exc = stmt.exc
        name = None
        if isinstance(exc, ast.Name):
            name = exc.id
        elif isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
            name = exc.func.id
        if name == "NotImplementedError":
            return True
    return False


def check_stub_density(
    workdir: Path, package: str
) -> tuple[bool, float, list[str]]:
    """Returns (passes, density, examples).

    `passes` is False iff the fraction of stub function bodies across
    NEW .py files in `{package}/` ≥ STUB_DENSITY_DOWNGRADE_THRESHOLD.
    Modified existing files aren't included — the wiring edits there
    are small by design.
    """
    pkg_prefix = f"{package}/"
    new_files = [
        workdir / p for p in changed_files(workdir)
        if p.startswith(pkg_prefix)
        and p.endswith(".py")
        and _file_is_new(workdir, p)
    ]
    if not new_files:
        return True, 0.0, []

    stub_count = 0
    total = 0
    examples: list[str] = []
    for fp in new_files:
        try:
            tree = ast.parse(fp.read_text(), filename=str(fp))
        except (SyntaxError, OSError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                total += 1
                if _is_stub_body(node):
                    stub_count += 1
                    if len(examples) < 5:
                        examples.append(f"{fp.name}:{node.name}")
    if total == 0:
        return True, 0.0, []
    density = stub_count / total
    return (density < STUB_DENSITY_DOWNGRADE_THRESHOLD, density, examples)


def check_tests_touch_existing_modules(
    workdir: Path, package: str
) -> tuple[bool, list[str]]:
    """If new package modules were added, at least one new test file must
    import from a non-new module in `{package}/`. Pure self-tests of the
    new file don't prove integration.

    No new package modules → vacuously passes (the integration is
    edits-only and the regular pytest gate is sufficient).

    Returns (passed, [example_existing_imports_seen]).
    """
    paths = changed_files(workdir)
    pkg_prefix = f"{package}/"
    new_pkg_files = [
        p for p in paths
        if p.startswith(pkg_prefix) and p.endswith(".py") and _file_is_new(workdir, p)
    ]
    if not new_pkg_files:
        return True, []

    new_pkg_stems = {Path(p).stem for p in new_pkg_files}

    # The gate also passes when the new capability is wired into the
    # package's existing surface — a pre-existing (non-test) package module
    # edited in this run imports the new module (e.g. a new exported nn
    # layer registered in `__init__.py`, or called by an existing module).
    # That is a genuine public-API integration even when the new test only
    # exercises the new module directly, so it shouldn't be demoted to an
    # Issue. (check_integration already proved the new code is invoked.)
    edited_existing_pkg = [
        workdir / p for p in paths
        if p.startswith(pkg_prefix) and p.endswith(".py")
        and not _file_is_new(workdir, p)
    ]
    for ef in edited_existing_pkg:
        try:
            tree = ast.parse(ef.read_text(), filename=str(ef))
        except (SyntaxError, OSError):
            continue
        referenced: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module:
                    referenced.add(node.module.rsplit(".", 1)[-1])
                for a in node.names:
                    referenced.add(a.name.rsplit(".", 1)[-1])
            elif isinstance(node, ast.Import):
                for a in node.names:
                    referenced.add(a.name.rsplit(".", 1)[-1])
        if referenced & new_pkg_stems:
            return True, [f"wired into existing module {ef.name}"]

    new_test_files = [
        workdir / p for p in paths
        if p.startswith("tests/") and p.endswith(".py") and _file_is_new(workdir, p)
    ]
    if not new_test_files:
        return False, []

    existing_imports: list[str] = []
    for tf in new_test_files:
        try:
            tree = ast.parse(tf.read_text(), filename=str(tf))
        except (SyntaxError, OSError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and (
                    node.module == package or node.module.startswith(f"{package}.")
                ):
                    rest = node.module[len(package):].lstrip(".")
                    head = rest.split(".")[0] if rest else ""
                    if head and head not in new_pkg_stems:
                        existing_imports.append(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == package or alias.name.startswith(f"{package}."):
                        rest = alias.name[len(package):].lstrip(".")
                        head = rest.split(".")[0] if rest else ""
                        if head and head not in new_pkg_stems:
                            existing_imports.append(alias.name)
    return (bool(existing_imports), existing_imports[:5])


def _classify_pytest(returncode: int, output: str) -> str:
    """Map a pytest run to "passed" | "failed" | "unvalidated".

    "unvalidated" means pytest could not actually exercise the change — no
    tests were collected (exit 5) or collection blew up on a missing
    dependency / import error. CI runners commonly install pytest but not
    the target repo's full dependency set (torch, tensorboard, …), so a
    collection ImportError is an environment limitation, NOT a code failure,
    and must not be reported as one.
    """
    if returncode == 0:
        return "passed"
    low = output.lower()
    real_failure = (
        " failed" in low
        or "assertionerror" in low
        or "= failures =" in low
    )
    if real_failure:
        return "failed"
    if returncode == 5:                       # no tests collected
        return "unvalidated"
    collection_markers = (
        "modulenotfounderror",
        "importerror",
        "error during collection",
        "errors during collection",
        "interrupted:",
    )
    if any(m in low for m in collection_markers):
        return "unvalidated"
    return "failed"


def run_tests(workdir: Path, timeout_s: int = 300) -> tuple[str, str]:
    """Run pytest. Returns (status, output) where status is one of
    "passed" | "failed" | "unvalidated" (see _classify_pytest)."""
    log.info(f"  → running pytest in {workdir}")
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", "-q", "--maxfail=3"],
            cwd=workdir, capture_output=True, text=True, timeout=timeout_s,
        )
        output = (result.stdout or "") + ("\n--- STDERR ---\n" + result.stderr if result.stderr else "")
        return _classify_pytest(result.returncode, output), output[-3000:]
    except subprocess.TimeoutExpired:
        return "failed", f"pytest timed out after {timeout_s}s"
    except Exception as e:
        return "failed", f"pytest invocation failed: {e}"


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
    payload = {"title": title, "body": full_body}
    try:
        issue = gh_api("POST", f"/repos/{target.repo}/issues", payload)
    except RuntimeError as e:
        # GitHub disables the Issues tab on forks (and some repos) by
        # default → POST /issues returns HTTP 410 "Issues has been
        # disabled". Enable it and retry once rather than failing the run.
        msg = str(e)
        if "Issues has been disabled" in msg or "HTTP 410" in msg:
            log.warning("  Issues disabled on repo; enabling and retrying")
            gh_api("PATCH", f"/repos/{target.repo}", {"has_issues": True})
            issue = gh_api("POST", f"/repos/{target.repo}/issues", payload)
        else:
            raise
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


# ─── Downgrade-to-Issue helper ─────────────────────────────────────────────


def _open_downgrade_issue(
    target: Target, rec: Recommendation, reason: str, detail: str,
) -> str:
    """Open an Issue when an automated post-implementation gate downgrades
    a PR-candidate to Issue. Used for the integration / stub-density /
    test-integration / self-review-can-delete branches in process_target.

    The body explains both *why this paper is interesting* (so the team
    keeps the discovery signal) and *why we didn't open a PR* (so the
    routing decision is auditable).
    """
    title = f"{PR_TITLE_PREFIX} {rec.paper_title}"
    body = (
        f"**Recommended paper**: "
        f"[{rec.paper_title}](https://arxiv.org/abs/{rec.arxiv_id})\n"
        f"**Confidence**: {rec.tier} "
        f"(Remyx relevance {rec.relevance_score:.2f})\n"
        f"**Research interest**: {rec.interest_name or '(unnamed)'}\n"
        f"\n---\n\n"
        f"## Why this paper is interesting for the team\n\n"
        f"{rec.reasoning or '(no reasoning provided)'}\n\n"
        f"## Suggested experiment\n\n"
        f"{rec.suggested_experiment or '(none)'}\n\n"
        f"## Why the orchestrator opened an Issue instead of a PR\n\n"
        f"**{reason}**\n\n"
        f"{detail}\n"
    )
    return open_issue(target, title, body)


# ─── Main per-target loop ──────────────────────────────────────────────────


def process_target(target: Target) -> dict:
    """Run the full discovery + implementation loop for one target.
    Returns a status dict suitable for logging / Slack notify.

    Routing summary — every path leads to either a PR, an Issue, or a
    skip:

        skipped_low_confidence            — tier below min_confidence
        skipped_rate_limit                — recent PR within rate-limit-days
        skipped_pr_exists                 — every candidate already has an
                                            open PR (or a mix of open PRs/Issues)
        skipped_issue_exists              — every candidate already has an
                                            open Remyx Issue

        issue_opened_preflight            — pre-flight (§6) routed to Issue
                                            before invoking implementation
        issue_opened                      — Claude wrote OPEN_AS_ISSUE.md
        issue_opened_no_integration       — integration validator (§2): the
                                            diff adds code nothing invokes
        issue_opened_stub_density         — stub-density validator (§3) rejected
        issue_opened_no_test_integration  — test gate (§3) found no test that
                                            imports an existing module
        issue_opened_self_review          — self-review (§4): new code is an
                                            orphan, unreachable from production

        rejected_path_violations          — Claude touched out-of-bounds paths
        skipped_test_failure              — draft_mode=never and tests failed
        claude_failed                     — Claude CLI exited non-zero

        pr_opened / pr_opened_draft       — happy path
    """
    result: dict = {"repo": target.repo, "status": "unknown"}

    # 1. Rate-limit (per-repo) — cheapest gate, before any candidate work
    #    or checkout.
    if recent_pr_within_rate_limit(target):
        result["status"] = "skipped_rate_limit"
        return result

    # 2. Query the candidate pool over the lookback window (default: the
    #    past week). The old flow took only papers[0], wasting the
    #    lookback; we keep the whole pool so the selection pass can pick
    #    the most implementable candidate.
    candidates = query_remyx_candidates(target)
    result["candidates_returned"] = len(candidates)

    # 3. Per-candidate gates. Drop anything below the confidence tier or
    #    already in flight — an open PR for its branch, OR an open Remyx
    #    Issue for the paper. The Issue check matters with a longer
    #    lookback: a sticky top candidate that keeps routing to Issue would
    #    otherwise be re-selected every run and reopen a duplicate Issue.
    #    Treating an open Issue as "in flight" drops it from the pool so
    #    selection advances to the next-best candidate; closing the Issue
    #    makes the paper eligible again. Running this BEFORE the clone
    #    preserves the "don't check out the repo if nothing is actionable"
    #    optimization the single-pick flow had.
    min_required = TIER_RANK.get(target.min_confidence.lower(), 2)
    open_issues = open_remyx_issues(target)
    viable: list[Recommendation] = []
    dropped_low_conf = 0
    dropped_pr_exists = 0
    dropped_issue_exists = 0
    for c in candidates:
        if TIER_RANK.get(c.tier.lower(), 0) < min_required:
            dropped_low_conf += 1
            continue
        c_branch = f"{BRANCH_PREFIX}{c.arxiv_id or slugify(c.paper_title)}"
        if existing_pr_for(target, c_branch):
            dropped_pr_exists += 1
            continue
        if issue_for_paper(open_issues, c):
            dropped_issue_exists += 1
            continue
        viable.append(c)

    if not viable:
        # Nothing actionable. Prefer the most specific skip reason.
        if dropped_low_conf and not dropped_pr_exists and not dropped_issue_exists:
            result["status"] = "skipped_low_confidence"
            log.info(f"  ✗ no candidate at/above min {target.min_confidence}; "
                     f"skipping")
        elif dropped_issue_exists and not dropped_pr_exists:
            result["status"] = "skipped_issue_exists"
            log.info(f"  ✗ all {dropped_issue_exists} candidate(s) already "
                     f"have open Remyx Issues; skipping")
        else:
            # PR dedup, or a mix of open PRs and Issues.
            result["status"] = "skipped_pr_exists"
            log.info(f"  ✗ all candidates already in flight "
                     f"({dropped_pr_exists} open PRs, "
                     f"{dropped_issue_exists} open Issues); skipping")
        return result

    log.info(f"  ✓ {len(viable)} viable candidate(s) "
             f"(dropped {dropped_low_conf} low-confidence, "
             f"{dropped_pr_exists} open PRs, "
             f"{dropped_issue_exists} open Issues)")

    # 4. Workdir + selection. Clone first (the selection pass needs the
    #    repo's module layout), then let Claude pick the candidate most
    #    directly implementable against this repo. Selection only chooses
    #    WHICH paper — the PR-vs-Issue decision stays with the gates below.
    workdir = prepare_workdir(target)
    try:
        package = detect_package_name(workdir)
        log.info(f"  detected package: {package}")

        pinned_idx = None
        if target.pin_arxiv:
            pinned_idx = next(
                (i for i, c in enumerate(viable) if c.arxiv_id == target.pin_arxiv),
                None,
            )
            if pinned_idx is None:
                log.warning(f"  pin-arxiv {target.pin_arxiv!r} not in viable "
                            f"pool; falling back to the selection pass")
        if pinned_idx is not None:
            rec = viable[pinned_idx]
            result["selection_reasoning"] = (
                f"(pinned via pin-arxiv={target.pin_arxiv})"
            )
            log.info(f"  ✓ pinned candidate [{pinned_idx}] {rec.paper_title[:50]}…")
        else:
            selection = select_recommendation(workdir, package, viable)
            if selection is not None:
                rec = viable[selection["chosen_index"]]
                result["selection_reasoning"] = selection.get("reasoning", "")
                result["selection_rejected"] = selection.get("rejected", [])
            else:
                rec = viable[0]
                result["selection_reasoning"] = (
                    "(selection pass unavailable — used top-ranked candidate)"
                )
        result.update({
            "paper": rec.paper_title,
            "arxiv": rec.arxiv_id,
            "tier": rec.tier,
            "candidates_considered": len(viable),
        })
        log.info(f"  ✓ selected: [{rec.tier}] {rec.paper_title}")

        # 5. Spec bundle for the chosen candidate. Thread the selection
        # rationale through so pre-flight and the implementer evaluate the
        # same scoped framing the selection pass reasoned about.
        branch = f"{BRANCH_PREFIX}{rec.arxiv_id or slugify(rec.paper_title)}"
        write_spec_bundle(
            workdir, target, rec, package,
            selection_note=result.get("selection_reasoning", ""),
        )

        # 5.5. Pre-flight Issue routing (§6). Cheap Claude pass that
        # decides PR vs Issue before we spend the implementation budget.
        # Failures here fall through — they don't block the PR path.
        preflight = preflight_routing(workdir, package)
        result["preflight_decision"] = (
            preflight.get("decision") if preflight else "(skipped)"
        )
        if preflight and preflight.get("decision") == "ISSUE":
            issue_title_inner = (
                preflight.get("issue_title")
                or f"{rec.paper_title}: needs team discussion"
            )
            issue_body_inner = (
                preflight.get("issue_body")
                or preflight.get("reasoning")
                or ""
            )
            issue_url = _open_downgrade_issue(
                target, rec,
                reason="Pre-flight routed to Issue before implementation",
                detail=(
                    f"{issue_body_inner}\n\n"
                    f"_Pre-flight reasoning: "
                    f"{preflight.get('reasoning', '(none)')}_"
                ),
            )
            # Override the body title with the preflight's title — it's
            # more specific than the generic paper title.
            result["status"] = "issue_opened_preflight"
            result["issue_url"] = issue_url
            log.info(f"  ✓ issue_opened_preflight: {issue_url}")
            return result

        # 6. Claude Code
        ok, claude_log = invoke_claude_code(
            workdir, timeout_s=target.claude_timeout_s
        )
        result["claude_exit_ok"] = ok
        # Always retain the Claude log tail — useful for diagnosing
        # silent-success-but-broken-state outcomes (e.g. orphan branch,
        # missing files), not just hard failures.
        result["claude_log_tail"] = claude_log[-1000:]
        if not ok:
            result["status"] = "claude_failed"
            return result

        # 6.5. Claude may have elected Issue-mode instead of writing code.
        issue_file = workdir / ISSUE_FALLBACK_FILENAME
        if issue_file.exists():
            log.info(f"  → Claude elected Issue-mode "
                     f"({ISSUE_FALLBACK_FILENAME} present); opening Issue")
            issue_title_inner, issue_body_inner = parse_issue_fallback_file(issue_file)
            issue_title = f"{PR_TITLE_PREFIX} {issue_title_inner}"
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

        # 7. Path allowlist enforcement.
        passed_allowlist, violations = validate_changes(workdir, target, package)
        if not passed_allowlist:
            result["status"] = "rejected_path_violations"
            result["violations"] = violations
            log.warning(f"  ✗ path violations: {violations}")
            return result

        # 7.5. Integration validator (§2). Rejects scaffold-shaped runs:
        # new module added with no existing-file edit referencing it,
        # too many new files, or oversized edits to existing files.
        passed_integration, int_violations = check_integration(
            workdir, target, package
        )
        if not passed_integration:
            result["integration_violations"] = int_violations
            log.warning(f"  ✗ integration check failed: {int_violations}")
            issue_url = _open_downgrade_issue(
                target, rec,
                reason="No real integration with the existing codebase",
                detail=(
                    "The implementation either added new modules without "
                    "wiring them into an existing call site, added too many "
                    "new files, or rewrote an existing file too aggressively. "
                    "Specifics:\n\n"
                    + "\n".join(f"- {v}" for v in int_violations)
                ),
            )
            result["status"] = "issue_opened_no_integration"
            result["issue_url"] = issue_url
            return result

        # 7.6. Stub density (§3). Routes to Issue if the new module's
        # public surface is dominated by pass / NotImplementedError /
        # empty bodies — i.e. the paper's contribution isn't really
        # present.
        density_ok, density, stub_examples = check_stub_density(workdir, package)
        result["stub_density"] = density
        if not density_ok:
            log.warning(
                f"  ✗ stub density {density:.0%} ≥ "
                f"{STUB_DENSITY_DOWNGRADE_THRESHOLD:.0%}; downgrading to Issue"
            )
            issue_url = _open_downgrade_issue(
                target, rec,
                reason=(
                    f"New module is mostly unimplemented "
                    f"({density:.0%} of function bodies are stubs)"
                ),
                detail=(
                    "The orchestrator's coding agent produced a module "
                    "whose public surface is dominated by `pass`, "
                    "`raise NotImplementedError`, or docstring-only "
                    "bodies. This usually means the paper's primary "
                    "contribution requires infra the repo doesn't have, "
                    "or there's no clear call site to extend.\n\n"
                    "Examples of stub bodies in the draft:\n\n"
                    + "\n".join(f"- `{e}`" for e in stub_examples)
                ),
            )
            result["status"] = "issue_opened_stub_density"
            result["issue_url"] = issue_url
            return result

        # 8. Tests
        tests_status, test_output = run_tests(workdir)
        result["tests_status"] = tests_status
        tests_passed = tests_status == "passed"
        result["tests_passed"] = tests_passed

        # 8.5. Test-touches-existing-modules gate (§3). If new package
        # modules were added, at least one new test must import from a
        # non-new module in the package — otherwise tests are pure
        # self-tests and don't prove integration.
        tests_touch_existing, existing_imports = (
            check_tests_touch_existing_modules(workdir, package)
        )
        result["tests_touch_existing"] = tests_touch_existing
        if not tests_touch_existing:
            log.warning(
                "  ✗ no new test imports from an existing module — "
                "tests only self-test the new file"
            )
            issue_url = _open_downgrade_issue(
                target, rec,
                reason=(
                    "New tests don't touch any pre-existing module"
                ),
                detail=(
                    "A new module was added, but none of the new test "
                    "files import from a pre-existing module in "
                    f"`{package}/`. Pure self-tests of the new file "
                    "don't prove the integration runs against existing "
                    "pipeline outputs."
                ),
            )
            result["status"] = "issue_opened_no_test_integration"
            result["issue_url"] = issue_url
            return result

        # 9. Self-review (§4). Second Claude pass over the diff. Renders
        # a "What this PR actually does" section into the PR body; if the
        # new code is an orphan (unreachable from any production path), it
        # routes to Issue. This is a REACHABILITY check, not a triviality
        # one — stub density (§3) already covers "the code is too thin".
        review = self_review_diff(workdir)
        result["self_review"] = review or {}
        if review and review.get("is_orphan") is True:
            log.warning(
                "  ✗ self-review: new code is unreachable from any "
                "production path (orphan); downgrading to Issue"
            )
            summary = review.get("honest_summary") or ""
            issue_url = _open_downgrade_issue(
                target, rec,
                reason="Self-review judged the new code an orphan (no production call path)",
                detail=(
                    "On a second pass over the diff, the coding agent "
                    "concluded that no pre-existing entry point or module "
                    "invokes the new code — at most its own tests call it. "
                    "That's an orphan: the product never exercises it. "
                    "(This is about reachability, not whether the code is "
                    "trivial — stub density is judged separately.)\n\n"
                    f"_Self-review summary: {summary}_"
                ),
            )
            result["status"] = "issue_opened_self_review"
            result["issue_url"] = issue_url
            return result
        review_section = _render_self_review_section(review) if review else ""

        # 10. Draft determination. "unvalidated" (tests couldn't run in CI,
        # e.g. the runner lacks the repo's deps) is NOT a failure: never-mode
        # opens a draft rather than skipping, and on_test_failure drafts it.
        if target.draft_mode == "always":
            draft = True
        elif target.draft_mode == "never":
            if tests_status == "failed":
                result["status"] = "skipped_test_failure"
                result["test_output_tail"] = test_output[-500:]
                return result
            draft = tests_status != "passed"
        else:                                # "on_test_failure"
            draft = tests_status != "passed"

        # 11. Commit + push + PR
        pr_title = f"{PR_TITLE_PREFIX} {rec.paper_title}"
        pr_body = build_pr_body(
            target, rec, tests_status, test_output,
            review_section=review_section,
            selection_note=result.get("selection_reasoning", ""),
        )
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


def build_pr_body(
    target: Target,
    rec: Recommendation,
    tests_status: str,
    test_output: str,
    review_section: str = "",
    selection_note: str = "",
) -> str:
    tier_emoji = {"high": "🟢", "moderate": "🟡", "low": "🟠", "noise": "🔴"}.get(rec.tier, "⚪")
    if tests_status == "passed":
        test_section_inner = "### Test results\n\n✅ All tests passed.\n"
    elif tests_status == "unvalidated":
        test_section_inner = (
            "### Test results\n\nℹ️ Tests could not run in CI — the runner "
            "lacks this repo's dependencies (a collection/import error, not "
            "a code failure). Run the suite locally to validate.\n\n"
            f"```\n{test_output[-1000:]}\n```\n"
        )
    else:
        test_section_inner = (
            "### Test results\n\n⚠️ Tests did not pass. PR opened as draft "
            f"for review.\n\n```\n{test_output[-1000:]}\n```\n"
        )
    # Self-review section (§4) goes ABOVE the test section so reviewers
    # see "what this PR actually does vs. what's stubbed" before the
    # green checkmark.
    test_section = (
        f"{review_section}\n{test_section_inner}"
        if review_section else
        test_section_inner
    )
    # Selection rationale: why this candidate was picked from the lookback
    # pool over higher-ranked ones. Empty (just the section break) when the
    # pool had one candidate or the selection pass was unavailable.
    selection_section = (
        f"\n## Why this candidate (selected from the lookback pool)\n\n"
        f"{selection_note}\n"
        if selection_note and not selection_note.startswith("(")
        else "\n"
    )
    return _PR_BODY_TEMPLATE.format(
        paper_title=rec.paper_title,
        arxiv_id=rec.arxiv_id,
        tier_emoji=tier_emoji,
        tier=rec.tier,
        relevance_score=rec.relevance_score,
        interest_name=rec.interest_name or "(unnamed)",
        reasoning=rec.reasoning or "(no reasoning provided)",
        selection_section=selection_section,
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

    timeout_raw = _optional_env("INPUT_CLAUDE_TIMEOUT", "900")
    try:
        claude_timeout_s = int(timeout_raw)
    except ValueError:
        log.error(f"INPUT_CLAUDE_TIMEOUT={timeout_raw!r} is not an integer.")
        sys.exit(2)

    return Target(
        repo=repo,
        interest_id=interest_id,
        min_confidence=_optional_env("INPUT_MIN_CONFIDENCE", "moderate"),
        rate_limit_days=rate_limit_days,
        draft_mode=draft_mode,
        guardrails_allowlist=guardrails_allowlist,
        claude_timeout_s=claude_timeout_s,
        pin_arxiv=_optional_env("INPUT_PIN_ARXIV", ""),
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
                if "candidates_considered" in result:
                    f.write(f"candidates_considered={result['candidates_considered']}\n")
                if "selection_rejected" in result:
                    f.write(f"selection_rejected={len(result['selection_rejected'])}\n")
        except OSError as e:
            log.warning(f"Could not write to $GITHUB_OUTPUT: {e}")

    # Non-zero exit on genuine failures so the workflow step fails visibly
    # (a green run with no PR/Issue previously masked claude_failed). Issues,
    # skips, and PRs stay green.
    if result.get("status") in FAILURE_EXIT_STATUSES:
        sys.exit(1)


if __name__ == "__main__":
    main()
