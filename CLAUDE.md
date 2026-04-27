# CLAUDE.md — ArgoEBUSAnalysis Project Guide

## Project Mission
Testing the "Ocean Refugia / Stealth Warming" hypothesis: subsurface warming in the California
Current System transported by the California Undercurrent that has not yet surfaced.

Three depth layers via Argo float GPR (Kriging):
- **Skin Layer** (0–100m): atmospheric forcing dominates
- **Source Layer** (150–400m): Ekman upwelling source water — where "stealth heat" hides
- **Background Layer** (500–1000m): deep ocean baseline

Signal: Source Layer warming faster than Background.

---

## Workflow Principles

1. **Plan mode** for any task with 3+ steps or architectural decisions. Re-plan if derailed.
2. **Subagents** for research, exploration, parallel analysis. One task per subagent.
3. **Self-improvement**: after any correction, update `AE_claude_lessons.md` AND this file.
4. **Verification**: never mark done without running the code and confirming output.
5. **Elegance check**: for non-trivial changes, ask "is there a more elegant way?"
6. **Autonomous bug fixing**: identify logs/errors, resolve without hand-holding.
7. **Token Management**: When token usage approaches 95%, recommend pausing the current plan, recording recent actions to `AE_claude_recentactions.md`, and updating the not completed portions of the current session to the top of `AE_claude_todo.md`.
8. **Context Reset (/clear)**: Before recommending `/clear` (at 20-turn intervals), first update `argo_claude_actions/AE_claude_recentactions.md` with everything accomplished this session, and move any unfinished tasks to the top of `argo_claude_actions/AE_claude_todo.md`. No context is lost across the reset.

---

## Autonomy & Approval Protocol

**Claude may act autonomously on implementation only after a detailed plan has been explicitly approved by the user in full.**

### Before approval — always required:
- Present the complete plan step by step, in maximum detail. Every file to be touched, every function to be changed, every logic decision. No vagueness.
- Wait for the user to say the plan is approved before touching any code, config, or file.
- Even small sub-steps within an approved plan must be announced verbosely before execution: "I am now doing Step 2: modifying `foo.py:42` to change X because Y."

### Absolute hard stops — never do these without explicit per-instance permission:
- **Creating new git branches.**
- **Deleting any file or directory.**
- **Implementing any plan that has not been explicitly approved.**

### During execution of an approved plan:
- Narrate every intermediate step before taking it, even if it seems obvious.
- If something unexpected comes up mid-plan (file missing, logic doesn't match expectation, new decision needed), stop and report before proceeding.
- Do not batch steps silently. One step, announce it, execute it, confirm, then move to the next.

---

## Scientific Context

- **Gemini**: science planning and hypothesis. **Claude**: implementation and debugging. Don't second-guess the science; flag implementation risks.
- **Anisotropy Ratio** = `Lat_Scale / Lon_Scale`. < 1.0 = zonal/atm forcing. > 1.0 = meridional current. Should increase with depth.
- **RMSRE** target < 5%. **Std Z-Score** ideal = 1.0.
- **run_id naming**: `california_20150101_20151231_res0_5x0_5_t30_0_d0_100` — `d0_100` = Skin, `d150_400` = Source, `d500_1000` = Background.

---

## Repository Structure

See `ae_file_structure.txt` for the full layout.

Key directories:
- `ArgoEBUSCloud/ebus_core/` — core library (config, thermodynamics, GPR, plotting)
- `AEResults/aeplots/`, `aelogs/` — output PNGs, audit CSVs, CV pickles
- `argo_claude_actions/` — todo, recent actions, lessons

---

## Style Preferences

- Keep responses concise and direct.
- No trailing summaries of what was just done.
- No emojis unless the user uses them.
- When referencing code, use `file:line_number` format.
- Prefer editing existing files over creating new ones.
- Do NOT add docstrings, comments, or type annotations to code that wasn't changed.
- **Always write verbose comments for any code you write or modify.** Every function must have a header comment explaining what it does, why it exists, what inputs mean physically, and what the output represents. Inline comments should explain non-obvious logic step by step. Code can be brief; comments should be thorough. The goal is that a human reading the code cold can follow every decision.

---

## Hard-Won Rules

**Always use `ebus-cloud-env`**: `conda run -n ebus-cloud-env python <script>`. Base env lacks all packages.

**Pipeline tools share a common signature**: every analysis/diagnostic function must be importable with `(region, lat_step, lon_step, time_step, depth_range)` — matching `run_diagnostic_inspection()`. No module-level constants. Enables serial calls across regions/layers in one script.

**Completed tasks leave `AE_claude_todo.md`**: record them in `AE_claude_recentactions.md`. Todo = forward-looking only.

**`/interrupt` command**: type `/interrupt` at any time to halt execution and write a checkpoint to `argo_claude_actions/checkpoint.md`. Resumes cleanly in the next message.

**`AEResults/` is at `ArgoEBUSAnalysis/`**, not inside `ArgoEBUSCloud/`. Paths must traverse up: `os.path.join(base_dir, "..", "AEResults", ...)`.

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview` + `list_communities`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
|------|----------|
| `detect_changes` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context` | Need source snippets for review — token-efficient |
| `get_impact_radius` | Understanding blast radius of a change |
| `get_affected_flows` | Finding which execution paths are impacted |
| `query_graph` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes` | Finding functions/classes by name or keyword |
| `get_architecture_overview` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes` for code review.
3. Use `get_affected_flows` to understand impact.
4. Use `query_graph` pattern="tests_for" to check coverage.

<!-- dgc-policy-v11 -->
# Dual-Graph Context Policy

This project uses a local dual-graph MCP server for efficient context retrieval.

## MANDATORY: Always follow this order

1. **Call `graph_continue` first** — before any file exploration, grep, or code reading.

2. **If `graph_continue` returns `needs_project=true`**: call `graph_scan` with the
   current project directory (`pwd`). Do NOT ask the user.

3. **If `graph_continue` returns `skip=true`**: project has fewer than 5 files.
   Do NOT do broad or recursive exploration. Read only specific files if their names
   are mentioned, or ask the user what to work on.

4. **Read `recommended_files`** using `graph_read` — **one call per file**.
   - `graph_read` accepts a single `file` parameter (string). Call it separately for each
     recommended file. Do NOT pass an array or batch multiple files into one call.
   - `recommended_files` may contain `file::symbol` entries (e.g. `src/auth.ts::handleLogin`).
     Pass them verbatim to `graph_read(file: "src/auth.ts::handleLogin")` — it reads only
     that symbol's lines, not the full file.
   - Example: if `recommended_files` is `["src/auth.ts::handleLogin", "src/db.ts"]`,
     call `graph_read(file: "src/auth.ts::handleLogin")` and `graph_read(file: "src/db.ts")`
     as two separate calls (they can be parallel).

5. **Check `confidence` and obey the caps strictly:**
   - `confidence=high` -> Stop. Do NOT grep or explore further.
   - `confidence=medium` -> If recommended files are insufficient, call `fallback_rg`
     at most `max_supplementary_greps` time(s) with specific terms, then `graph_read`
     at most `max_supplementary_files` additional file(s). Then stop.
   - `confidence=low` -> Call `fallback_rg` at most `max_supplementary_greps` time(s),
     then `graph_read` at most `max_supplementary_files` file(s). Then stop.

## Token Usage

A `token-counter` MCP is available for tracking live token usage.

- To check how many tokens a large file or text will cost **before** reading it:
  `count_tokens({text: "<content>"})`
- To log actual usage after a task completes (if the user asks):
  `log_usage({input_tokens: <est>, output_tokens: <est>, description: "<task>"})`
- To show the user their running session cost:
  `get_session_stats()`

Live dashboard URL is printed at startup next to "Token usage".

## Rules

- Do NOT use `rg`, `grep`, or bash file exploration before calling `graph_continue`.
- Do NOT do broad/recursive exploration at any confidence level.
- `max_supplementary_greps` and `max_supplementary_files` are hard caps - never exceed them.
- Do NOT dump full chat history.
- Do NOT call `graph_retrieve` more than once per turn.
- After edits, call `graph_register_edit` with the changed files. Use `file::symbol` notation (e.g. `src/auth.ts::handleLogin`) when the edit targets a specific function, class, or hook.

## Context Store

Whenever you make a decision, identify a task, note a next step, fact, or blocker during a conversation, call `graph_add_memory`.

**To add an entry:**
```
graph_add_memory(type="decision|task|next|fact|blocker", content="one sentence max 15 words", tags=["topic"], files=["relevant/file.ts"])
```

**Do NOT write context-store.json directly** — always use `graph_add_memory`. It applies pruning and keeps the store healthy.

**Rules:**
- Only log things worth remembering across sessions (not every minor detail)
- `content` must be under 15 words
- `files` lists the files this decision/task relates to (can be empty)
- Log immediately when the item arises — not at session end

## Session End

When the user signals they are done (e.g. "bye", "done", "wrap up", "end session"), proactively update `CONTEXT.md` in the project root with:
- **Current Task**: one sentence on what was being worked on
- **Key Decisions**: bullet list, max 3 items
- **Next Steps**: bullet list, max 3 items

Keep `CONTEXT.md` under 20 lines total. Do NOT summarize the full conversation — only what's needed to resume next session.
