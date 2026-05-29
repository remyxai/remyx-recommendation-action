# Remyx Recommendation — GitHub Action

A composite GitHub Action that, on a schedule you choose, picks the next arXiv paper for your team to integrate (via the Remyx engine API) and either opens a draft pull request that wires the paper's contribution into an existing call site in your codebase, or opens a discussion Issue when the paper doesn't fit.

```yaml
- uses: remyxai/remyx-recommendation-action@v1
  with:
    interest-id: ${{ vars.REMYX_INTEREST_ID }}
```

## What you get

Each scheduled run:

1. Queries `engine.remyx.ai` for the candidate pool against your team's configured `ResearchInterest` over the `lookback` window (default: the past week), then runs a Claude **selection pass** that picks the candidate most directly implementable against *your* repo. Relevance rank alone often surfaces a model-architecture or training-method paper with no call site in a data/inference pipeline, while a lower-ranked candidate is a clean drop-in — the selection pass reads your repo's module layout and chooses accordingly (Remyx already knows your repo's commit history and codifies what your team has been building).
2. Either:
   - Opens a **draft pull request** that adds a small capability-named module AND wires it into an existing call site in your package, with a self-review section in the PR body honestly noting what was implemented vs. left out, OR
   - Opens an **Issue** with the paper's details and a discussion of what would block a clean integration — when the paper doesn't fit (pre-flight, validators, or self-review route it to discussion instead of a PR).
3. Reports outputs (`status`, `pr_url`, `issue_url`, `arxiv`, `tier`) you can chain into downstream steps.

The orchestrator defaults to opening Issues, not PRs. A PR is only opened when the implementation wires new code into an existing module, passes a stub-density check, has at least one test that imports from a pre-existing module, and survives a self-review pass over the diff. This makes PRs ready-to-ship rather than scaffold-shaped.

The selection pass only chooses *which* candidate to implement — it never decides PR vs Issue. The selected candidate still runs the full pre-flight + integration / stub / test / self-review gate chain, so if even the best-fit candidate can't be cleanly implemented, it's routed to an Issue exactly as before.

## Setup (5 minutes)

### 1. Configure your interest at engine.remyx.ai

Sign up at [engine.remyx.ai](https://engine.remyx.ai). Connect your GitHub repo — Remyx ingests your commit history and creates a `ResearchInterest` that captures your team's research focus. Edit the interest's context body to sharpen the framing.

### 2. Get your `REMYX_API_KEY`

From the engine.remyx.ai Settings page, generate a long-lived API key for this repo's automation. Copy it.

### 3. Add secrets to your repo

In your repo's **Settings → Secrets and variables → Actions**, add:

| Secret | Source |
|---|---|
| `REMYX_API_KEY` | the key from step 2 |
| `ANTHROPIC_API_KEY` | your Anthropic key from [console.anthropic.com](https://console.anthropic.com) — used to invoke Claude Code |

### 3b. Allow Actions to open pull requests

In **Settings → Actions → General → Workflow permissions**, enable:

> ☑ Allow GitHub Actions to create and approve pull requests

This is disabled by default at the org level for new repos. Without it, the action runs to completion but the final PR-creation step returns `HTTP 403: GitHub Actions is not permitted to create or approve pull requests`. The setting can be enabled per-repo without changing org defaults.

### 4. Add the workflow

Create `.github/workflows/remyx-recommendation.yml`:

```yaml
name: Remyx Recommendation

on:
  schedule:
    - cron: '0 6 * * 1'        # Mondays at 06:00 UTC; pick whatever cadence suits you
  workflow_dispatch:            # also lets you trigger manually

jobs:
  recommend:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      pull-requests: write
      issues: write
    env:
      # Inherited by the composite action's subprocesses.
      REMYX_API_KEY: ${{ secrets.REMYX_API_KEY }}
      ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
    steps:
      - uses: remyxai/remyx-recommendation-action@v1
        with:
          interest-id: 'YOUR-INTEREST-UUID-HERE'
```

Replace `YOUR-INTEREST-UUID-HERE` with the UUID from engine.remyx.ai. (Tip: the engine UI offers a "copy workflow snippet" button that emits this YAML pre-filled.)

### 5. First run

Visit your repo's **Actions** tab → **Remyx Recommendation** → **Run workflow**. The first run takes 4-6 minutes; subsequent scheduled runs are identical. A draft PR appears in your **Pull Requests** tab when it completes.

## Inputs

| Input | Default | Description |
|---|---|---|
| `interest-id` | *(required)* | Remyx ResearchInterest UUID. |
| `github-token` | *(empty — falls back to `${{ github.token }}`)* | Token used for git push + PR/Issue creation. Leave unset for the standard same-repo install; the action will use the workflow's built-in `GITHUB_TOKEN`. Override with a PAT (`${{ secrets.MY_PAT }}`) only for cross-repo controller patterns. |
| `min-confidence` | `moderate` | Tier gate: `high` / `moderate` / `low`. Recommendations below this are skipped. |
| `draft-mode` | `always` | PR-draft policy: `always` (default), `on_test_failure`, or `never`. |
| `rate-limit-days` | `7` | Skip if a previous Remyx PR was opened within this window. Set `0` to rely only on per-paper dedup. |
| `guardrails-allowlist` | `''` | Comma-separated extra path globs Claude Code may modify (most repos don't need this). |
| `lookback` | `week` | Recommendation lookback window: `today` / `week` / `month`. The candidate pool is pulled over this window before the selection pass. |
| `candidate-pool` | `25` | How many recommendations to pull into the selection pool. The selection pass picks the most implementable one; the rest are recorded as rejected with a reason. |

## Outputs

| Output | When set | Description |
|---|---|---|
| `status` | always | Run outcome (see status table below) |
| `pr_url` | when status starts with `pr_opened` | URL of the opened PR |
| `issue_url` | when status starts with `issue_opened` | URL of the opened Issue |
| `arxiv` | when a recommendation was fetched | arxiv_id of the picked paper |
| `tier` | when a recommendation was fetched | Confidence tier (`high` / `moderate` / `low` / `noise`) |

## Status codes

| Status | Meaning |
|---|---|
| `pr_opened` | PR opened ready-for-review (tests passed, `draft-mode != always`) |
| `pr_opened_draft` | PR opened as draft (the default under `draft-mode: always`, or when tests failed under `draft-mode: on_test_failure`) |
| `issue_opened_preflight` | Pre-flight Claude pass routed to Issue **before** spending the implementation budget (paper needs infra the repo lacks, or no clear call site) |
| `issue_opened` | Claude elected Issue-mode during implementation (wrote `OPEN_AS_ISSUE.md` instead of code) |
| `issue_opened_no_integration` | New module(s) added but no existing file was modified to wire them in — orphan scaffold detected |
| `issue_opened_stub_density` | New module's public surface is dominated by `pass` / `raise NotImplementedError` / empty bodies (≥50% of function bodies are stubs) |
| `issue_opened_no_test_integration` | New tests only self-test the new file; no new test imports from a pre-existing module |
| `issue_opened_self_review` | Second Claude pass over the diff concluded the changes are deletable with no functional loss |
| `skipped_low_confidence` | Recommendation below `min-confidence` |
| `skipped_rate_limit` | A previous Remyx PR was opened within `rate-limit-days` |
| `skipped_pr_exists` | An open PR for this exact paper already exists |
| `skipped_test_failure` | Tests failed AND `draft-mode: never` |
| `claude_failed` | Claude CLI exited non-zero |
| `rejected_path_violations` | Claude touched files outside the guardrails allowlist; no PR opened |
| `error` | Unhandled exception (action step exits 1) |

## Guardrails — what Claude can and can't modify

Allowed paths (defaults):
- `<package>/**/*.py` — any module in your target package (new or existing, so Claude can add the wiring edit at the call site)
- `tests/**/*.py` — any test file
- `.remyx-recommendation/**` — the spec bundle (scrubbed before commit, never lands in the PR)
- `README.md` — append-only attribution section

Always blocked:
- `.github/**`, `docker/**`, `pipelines/**`, `config/**`
- `requirements.txt`, `setup.py`, `pyproject.toml`, `MANIFEST.in`

Edit-size caps (post-hoc, enforced after the Claude session):
- Each edit to a pre-existing file is capped at **50 net lines** (additions + deletions). Larger edits get rejected — wiring is expected to be surgical.
- At most **3 new `.py` files** under the target package per run.

If Claude touches anything outside the allowed set, the action rejects the run and does not open a PR. Use the `guardrails-allowlist` input to extend the allowed set for your repo.

## How it works

```
GitHub cron fires the workflow
       ↓
GET engine.remyx.ai/api/v1.0/papers/recommended?interest_id=<id>
GET engine.remyx.ai/api/v1.0/research-interests/<id>     ← team focus context
       ↓
Confidence + dedup gates
       ↓
Clone repo, branch from main
       ↓
Write .remyx-recommendation/ spec bundle (briefing for Claude)
       ↓
Pre-flight Claude pass: PR or Issue?
       ↓                              ↓
     ISSUE                            PR
       ↓                              ↓
   open Issue        Invoke claude --dangerously-skip-permissions
                                      ↓
                     Either: open OPEN_AS_ISSUE.md (→ Issue mode)
                     or implement INTEGRATION (call-site edit + new module)
                                      ↓
                     Path-allowlist check  +  integration validator
                     (new module must be imported by a modified file)
                                      ↓
                     Stub-density check (new module not mostly TODOs)
                                      ↓
                     pytest  +  test-integration check
                     (≥1 new test must import a pre-existing module)
                                      ↓
                     Self-review pass over the diff
                     (PR body gets "What this PR actually does" section;
                      downgrade to Issue if diff is deletable with no loss)
                                      ↓
                     Commit (with bundle scrubbed) + push + open draft PR
```

The recommendation engine (commit-history extraction, candidate pool, embedding pre-filter, Gemini ranking) lives server-side on engine.remyx.ai. This action is a pure consumer; the API call returns a fully-formed recommendation with reasoning + suggested experiment + interest context body.

## Costs

Per run:
- Remyx API: included in your engine.remyx.ai subscription
- Claude Code: ~$0.40-0.70 per draft-PR run (you bring your own `ANTHROPIC_API_KEY`) — the pre-flight + implementation + self-review passes together. Runs routed to Issue at pre-flight skip the implementation pass and cost less.
- GitHub Actions minutes: ~5 minutes per run on `ubuntu-latest`

At weekly cadence (~4 runs/mo) with ~50% confidence-gate skip rate, expect ~$1-3/mo Claude + a handful of free-tier Actions minutes.

## License

Apache 2.0. See [LICENSE](./LICENSE).
