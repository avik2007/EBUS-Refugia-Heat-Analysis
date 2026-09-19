# Plan: Close out AWS/Coiled, clean up git, then resume spatial-window GP pilot

## Context
Tree is dirty (14 modified + 80 untracked, all uncommitted) on `fix/restore-antigravity-hln-vertical-delta`, a branch
whose name no longer matches its content (d_0 identifiability study). Several stale branches exist. User decisions this
turn: close out AWS/Coiled for now, drop the 32 remaining LML sweeps, do git cleanup first, then return to the
spatial-moving-window pilot (Part C, saved below unchanged in substance).

## Git state (verified read-only)
- `main` == current branch == 935ea59. `main` is 2 commits ahead of `origin/main` (a1a1c21, 935ea59, unpushed).
  Current branch is already on origin at 935ea59. So there is nothing to merge; only a push/PR question.
- Local branches:
  - `feat/github-pages-portfolio`: fully merged into main (PRs #5/#7/#8). Also on origin.
  - `feature/depth-aware-float-census`: merged into main, worktree `.worktrees/depth-aware-float-census` is CLEAN.
  - `fix/portfolio-url`: 1 patch NOT in main (b7d5749 "remove em-dashes, add Source layer anisotropy plot"); 41a1a91 is already in main by patch.
  - `main`, `fix/restore-antigravity-hln-vertical-delta`.
- Remote-only: `origin/gaussian-kriging-rework` (all patches in main), `origin/feat/mlops-phase2` (1 patch NOT in main: 4324fe8).
- Dirty tree groups:
  - Study code: `argoebus_gp_physics.py`, `runner.py`, `02_ae_cloud_run.py`, `05_ae_update_tomatern0.5.py`,
    `test_mlops_foundation.py`, `vertical_delta_analysis.py`, `d0_lml_sweep.py`, `d0_distribution_multiyear.py`,
    `summarize_audits.py`, 64 `configs/californiav3/*` (2010-2020 ingest + gibbs_lml yaml, 1 modified timelsfix yaml).
  - Docs/notes: `CLAUDE.md`, `argo_claude_actions/*`, `argo_gemini_actions/*`, `CONTEXT.md`, `References/*`,
    `docs/superpowers/*`, `docs/presentations/`, `docs/images/whoi_*.jpg`, `argo_probe_for_wherobots_int.md`, `.gitignore`.
  - Junk to never commit: `*:Zone.Identifier` files (Windows metadata), probably `.claude/settings.local.json`.

## Steps (each needs your OK where marked; nothing runs until approved)
A0. Close out AWS/Coiled (docs only): edit `argo_claude_actions/AE_claude_todo.md` to remove the [TOP] optimizer-test
    item and the 32-sweep item (move to `AE_claude_recentactions.md` as "deferred by user 2026-09-19"), update
    `CONTEXT.md` and memory `project_d0_identifiability.md`. No S3 prefix created, no Coiled call made.
A1. Read-only review of diffs (`git diff` of the 14 modified files) to confirm each belongs to the d_0 study and that
    no secrets/paths leak; inspect b7d5749 and 4324fe8 to decide keep/drop.
A2. Add `*:Zone.Identifier` (and `.claude/settings.local.json` if not already ignored) to `.gitignore`. Small edit.
A3. Commit grouping on the CURRENT branch (no new branch): (1) `.gitignore` chore; (2) d_0 study code + tests +
    configs; (3) docs/notes/references; (4) CONTEXT.md. Files staged by explicit path, never `git add -A`.
    Run pytest from repo root before commit (2). Co-Authored-By trailer per session instruction.
A4. [needs your OK] Publish: recommended = push current branch, open PR to main (matches your PR #5-#9 workflow), then
    local main fast-forwards after merge. Alternative = push main directly. Pushes are outward-facing.
A5. [needs per-branch OK, hard stop] Deletions after A4: local `feat/github-pages-portfolio`,
    `feature/depth-aware-float-census` (+ `git worktree remove` first, clean), `origin/gaussian-kriging-rework`;
    `fix/portfolio-url` and `origin/feat/mlops-phase2` only if A1 shows their unique patch is obsolete
    (otherwise cherry-pick or keep). Branch rename of the fix/... branch is optional and needs your OK
    (renaming = new branch name).

## Part C (deferred, resume after git is clean): spatial moving-window GP pilot
Same content as the plan you rejected, still valid: Source d150_400 / 2015 / californiav3, same 45d/10d time windows and
KFold(10%, seed 42); (1) heavy-tail diagnostic incl. saving CV residuals (`cv_details` pkl is currently an EMPTY dict);
(2) spatial-window driver with +-3-4 deg box (not Kuusela's 20 deg), stationary matern0.5/rbf per box, common signature
`(region, lat_step, lon_step, time_step, depth_range)`, no edits to Gibbs or existing validate functions; (3) same-CV
comparison vs Gibbs with existing DM/HLN tools + implied correlation at fixed lags; (4) coast-vs-offshore length-scale
map to decide whether d_0 can be fixed physically; (5) EM scale-mixture robust noise only if kurtosis warrants it.
Fact recorded: Gibbs spatial part is squared-exponential (`argoebus_gp_physics.py` ~296-298), header comment says
Matern-0.5-like (comment is wrong; fixing it is a separate approval).

## Verification
- After A3: `git status` clean except intentionally untracked; `git log --oneline` shows the grouped commits;
  pytest output attached. After A5: `git branch -a` and `git worktree list` show only intended refs.
