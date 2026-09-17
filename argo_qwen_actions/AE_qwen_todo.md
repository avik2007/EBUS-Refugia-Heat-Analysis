# AE_qwen_todo.md — Qwen forward-looking task queue

Qwen is the **implementation executor** (run via aider). Claude and Antigravity
write dated task briefs into `argo_qwen_actions/` as
`YYYY-MM-DD_HHMM_<slug>.md`. Newest brief with `Status: OPEN` is the next job.

After finishing a brief:
1. flip its header `Status:` to `DONE` (or `BLOCKED — <reason>`),
2. append a dated entry to `AE_qwen_recentactions.md`,
3. remove the item from this file,
4. record any correction/gotcha in `AE_qwen_lessons.md`.

---

## Open

_(none)_

## Blocked

_(none)_
