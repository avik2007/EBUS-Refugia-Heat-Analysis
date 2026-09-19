# Session Context (2026-09-19)

## Current Task
Git cleanup (commit dirty tree in groups, tidy stale branches), then spatial moving-window GP pilot.

## Key Decisions
- d_0 must become physical (coast vs gyre / undercurrent vs CCS); copying Kuusela & Stein 2018 (local stationary GP,
  spatial + temporal moving window). Stay Gaussian; Student-t Laplace is unstable, so check kurtosis first.
- AWS/Coiled optimizer test and the 32 remaining LML sweeps are DEFERRED (nothing launched, no S3 prefix).
- Gibbs is a signal kernel (spatial part squared-exponential); noise is the separate WhiteKernel.

## Next Steps
- Plan: `.claude/plans/we-need-to-look-mutable-rabbit.md` (Part A git steps A1-A5, then Part C pilot).
- Branch `fix/restore-antigravity-hln-vertical-delta` == main; nothing from the d_0 study is committed yet.

Details: `argo_claude_actions/AE_claude_recentactions.md` (2026-09-19).
