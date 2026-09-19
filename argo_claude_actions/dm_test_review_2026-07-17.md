# Report: Diebold-Mariano significance test review (Gibbs vs Matérn)

**Date:** 2026-07-17
**Script under review:** `compare_kernels.py` (written by Gemini, session 2026-07-17)
**Status:** Reviewed, gap found, fix NOT yet implemented (pending approval)

## Background

Session 18 (2026-07-07) settled the methodology for judging whether Gibbs's RMSRE
improvement over Matérn is real vs. noise from autocorrelated overlapping windows:
Diebold-Mariano test, HAC/Newey-West variance, **truncation lag = window/step ratio − 1**,
**small-sample Harvey-Leybourne-Newbold (HLN) correction**. Recorded in
`AE_claude_lessons.md` #5 and `AE_gemini_lessons.md`.

Gemini implemented `compare_kernels.py` this session and logged the DM audit as
complete in `AE_gemini_recentactions.md` (2026-07-17 entry), reporting statistically
significant Gibbs superiority on all 3 layers.

## What I verified

Ran `compare_kernels.py` directly (`conda run -n ebus-cloud-env python compare_kernels.py`).
No errors. Output matches the numbers already logged in `AE_gemini_recentactions.md` exactly:

| Layer | Rel. median RMSRE improvement | DM stat | DM p (one-sided) |
|---|---|---|---|
| Skin | 12.86% | 3.193 | 7.05e-04 |
| Source | 13.53% | 2.051 | 2.01e-02 |
| Background | 18.87% | 5.068 | 2.01e-07 |

Z-score calibration numbers also match prior verification (session 19/20): Gibbs
mean~0.97-1.00, std 0.083-0.100 vs Matérn mean 1.13-1.72, std 1.05-2.63.

## Gap found

`compare_kernels.py` implements HAC/Newey-West variance but **not** the second half of
the agreed methodology:

1. **No HLN small-sample correction.** p-values are computed against a standard normal
   distribution (`stats.norm.cdf`). With N=34 matched windows, the DM test's asymptotic-
   normal approximation is known to be anti-conservative — HLN (1997) corrects this by
   scaling the DM statistic down and comparing against Student's t(N-1) instead of normal.
   Without it, the reported p-values are likely too optimistic (more "significant" than
   they should be).
2. **Lag hardcoded to 4, not derived.** The agreed rule was `lag = window/step ratio − 1`.
   For this pipeline: `window_size_days=45`, `step_size_days=10` → `45/10 − 1 = 3.5`.
   The script uses `lag=4` (a plausible rounding, but not computed from config — if
   `window_size_days`/`step_size_days` change in future runs, this script silently goes
   stale).

Neither issue is a crash or a wrong-code bug — the script runs and produces internally
consistent, correctly-computed DM statistics under the (incomplete) formula it uses. The
gap is that it stops short of the full methodology that was explicitly agreed upon,
which is likely why Gemini's log calls it a completed "audit" while the underlying task
in `AE_gemini_todo.md` / session 18 notes describes HLN as part of the plan.

## Recommendation (not yet actioned)

Add the HLN correction to `dm_test()`:
- `h` = forecast horizon, analogous to `lag + 1` here (so `h=5` given `lag=4`, or derive
  both from config).
- `DM* = DM * sqrt((N + 1 - 2h + h*(h-1)/N) / N)`
- Compare `DM*` against `t(N-1)` (via `stats.t.cdf`), not `stats.norm.cdf`.

Then re-run and check whether the "statistically superior" verdict survives for all 3
layers, especially **Source** (current one-sided p=2.01e-02 is the closest to the 0.05
line and has the widest block-bootstrap CI already crossing near zero:
`[-0.00008, 0.00510]`) — this is the layer most at risk of losing significance under the
more conservative correction.

## Next step

**Resume next session** (see top of `AE_claude_todo.md`): implement the HLN correction
and derive lag from config, re-run `compare_kernels.py`, and confirm/update the verdict
before this significance claim is used anywhere external-facing (LinkedIn post,
interview prep, Gemini briefing).
