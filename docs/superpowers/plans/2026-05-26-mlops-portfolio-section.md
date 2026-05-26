# MLOps Portfolio Section Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a single "Engineering Infrastructure" section to `docs/index.html` advertising the MLOps CLI mechanisms (config schema, manifests, collision detection, CLI workflow).

**Architecture:** Insert one `<section>` block between the "Baseline Results" section (ends line 296) and the "Key Finding" section (starts line 298) of `docs/index.html`. Reuse all existing CSS classes; add one small `<style>` block for the code block dark background.

**Tech Stack:** HTML, Bootstrap 5.3, existing CSS custom properties

---

### Task 1: Add pre/code styling and insert the MLOps section

**Files:**
- Modify: `docs/index.html` lines 80–81 (add `<pre>` style before `</style>`) and lines 296–297 (insert section)

- [ ] **Step 1: Add `<pre>` code block style**

In `docs/index.html`, find this line (currently line 80):

```
    footer { padding: 2.5rem 0; border-top: 1px solid var(--border); text-align: center; color: var(--muted); font-size: .875rem; background: var(--surface); }
```

Insert the following line immediately before it:

```css
    /* Code block */
    .code-wrap { background: #0f1923; border: 1.5px solid var(--border); border-radius: 14px; padding: 1.5rem; overflow-x: auto; }
    .code-wrap pre { margin: 0; color: #c9d1d9; font-size: .82rem; line-height: 1.7; font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; }
    .code-wrap .cm { color: #8b949e; }
```

- [ ] **Step 2: Insert the MLOps section between Results and Key Finding**

In `docs/index.html`, find this exact block (lines 296–298):

```html
</section>

<!-- Key Finding -->
```

Replace it with:

```html
</section>

<!-- MLOps Infrastructure -->
<section>
  <div class="container">
    <p class="section-label">Engineering Infrastructure</p>
    <h2 class="section-title">Config-Driven Pipeline with Reproducibility Guarantees</h2>
    <p class="section-sub">Every run is schema-validated, content-addressed, and registered. YAML config to signed manifest in one command.</p>

    <div class="row mt-4 g-3">
      <div class="col-md-4">
        <div class="card p-3 h-100">
          <p style="font-weight:800;margin-bottom:.4rem;font-size:.95rem">Config-Driven Runs</p>
          <p class="mb-0" style="font-size:.88rem;color:var(--muted);line-height:1.6">
            <strong style="color:var(--text)">Pydantic v2 schema</strong> with strict validation
            (<code>extra="forbid"</code>), a <code>schema_version</code> field, and cross-field validators
            that prevent parameter aliasing. YAML configs live in <code>configs/&lt;region&gt;/</code>
            — no script editing needed to run a new configuration.
          </p>
        </div>
      </div>
      <div class="col-md-4">
        <div class="card p-3 h-100">
          <p style="font-weight:800;margin-bottom:.4rem;font-size:.95rem">Immutable Manifests</p>
          <p class="mb-0" style="font-size:.88rem;color:var(--muted);line-height:1.6">
            <strong style="color:var(--text)">SHA-256 hash of the canonical config</strong> is the run
            identity. Every manifest captures the git commit SHA, conda environment, hostname, and
            wall-clock duration at the moment of execution. Written once, never mutated.
          </p>
        </div>
      </div>
      <div class="col-md-4">
        <div class="card p-3 h-100">
          <p style="font-weight:800;margin-bottom:.4rem;font-size:.95rem">Run Registry + Collision Guard</p>
          <p class="mb-0" style="font-size:.88rem;color:var(--muted);line-height:1.6">
            <strong style="color:var(--text)">JSONL ledger</strong> of all completed runs, queryable
            by region or kind. The collision detector blocks re-running an identical config before
            compute is spent — <code>--force-overwrite</code> required to override.
          </p>
        </div>
      </div>
    </div>

    <div class="code-wrap mt-4">
      <pre><span class="cm"># validate before committing compute</span>
aebus validate configs/california/california_20150101_20151231_res0_5x0_5_t30_0_d150_400_3dmatern_w45.yaml

<span class="cm"># run the full GPR pipeline</span>
aebus analyze configs/california/california_20150101_20151231_res0_5x0_5_t30_0_d150_400_3dmatern_w45.yaml

<span class="cm"># list all completed Source-layer runs</span>
aebus list --region california --kind analysis

<span class="cm"># inspect provenance for any run</span>
aebus show california_20150101_20151231_res0_5x0_5_t30_0_d150_400_3dmatern_w45</pre>
    </div>

    <div class="mt-3 text-center">
      <span class="stat-chip"><span class="val">54</span> tests</span>
      <span class="stat-chip"><span class="val">Pydantic v2</span> schema</span>
      <span class="stat-chip"><span class="val">SHA-256</span> content addressing</span>
      <span class="stat-chip"><span class="val">JSONL</span> run registry</span>
    </div>
  </div>
</section>

<!-- Key Finding -->
```

- [ ] **Step 3: Verify no em-dashes introduced**

```bash
grep -c "—" docs/index.html
```

Expected output: `0`

- [ ] **Step 4: Verify the new section is present**

```bash
grep -c "Engineering Infrastructure\|collision\|aebus validate\|aebus analyze\|aebus list\|aebus show" docs/index.html
```

Expected output: `6`

- [ ] **Step 5: Commit**

```bash
git add docs/index.html
git commit -m "feat(portfolio): add MLOps infrastructure section with CLI showcase"
```

---

### Task 2: Push and merge to main

- [ ] **Step 1: Push to remote branch**

```bash
git push origin HEAD:feat/github-pages-portfolio
```

- [ ] **Step 2: Open and merge PR (or push directly to main if branch is clean)**

```bash
gh pr create --base main --head feat/github-pages-portfolio \
  --title "feat(portfolio): add MLOps infrastructure section" \
  --body "Adds Engineering Infrastructure section with 3 capability cards, CLI workflow showcase, and stat chips."
gh pr merge --merge
```

