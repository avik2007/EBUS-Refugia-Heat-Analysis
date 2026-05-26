# Spec: MLOps Portfolio Section

**Date:** 2026-05-26
**Status:** Approved

---

## Goal

Add one new section to `docs/index.html` advertising the MLOps CLI infrastructure built for the ArgoEBUS pipeline. Audience: ML hiring managers (scan code) and scientist collaborators (read cards).

---

## Position

Insert between the existing "Baseline Results" section and the "Key Finding" section.

---

## Section Structure

### Header
- Section label: `Engineering Infrastructure`
- H2: `Config-Driven Pipeline with Reproducibility Guarantees`
- Section-sub: `Every run is schema-validated, content-addressed, and registered — from YAML to manifest in one command.`

### Capability Cards (row of 3, col-md-4 each)

1. **Config-Driven Runs**
   - Pydantic v2 schema with strict validation (`extra="forbid"`)
   - `schema_version` field — bumping requires a documented migration
   - Cross-field validators prevent bin-aliasing and illegal parameter combos
   - YAML configs in `configs/<region>/` — no script editing needed

2. **Immutable Manifests**
   - SHA-256 hash of canonical config = run identity
   - Captures: git commit SHA, conda environment, hostname, wall-clock duration
   - Manifest is written once and never mutated
   - Content-addressed: two configs that differ only in `description` produce the same hash

3. **Run Registry + Collision Guard**
   - JSONL ledger (`AEResults/run_registry.jsonl`) of all completed runs
   - Collision detector blocks re-running an identical config before compute is spent
   - `--force-overwrite` flag available to explicitly override

### CLI Showcase (code block)

```bash
# validate before committing compute
aebus validate configs/california/source_layer.yaml

# run the full GPR pipeline
aebus analyze configs/california/source_layer.yaml

# list all completed runs for a region
aebus list --region california --kind analysis

# inspect provenance for any run
aebus show california_20150101_20151231_res0_5x0_5_t30_0_d150_400_3dmatern_w45
```

### Stat Chips Row

`54 tests` · `Pydantic v2` · `SHA-256 content addressing` · `JSONL registry`

---

## Implementation Notes

- Use existing CSS classes: `section-label`, `section-title`, `section-sub`, `card`, `stat-chip`
- Code block: wrap in `<pre><code>` inside an `arch-img-wrap`-style container with dark background
- No new CSS needed beyond a `<pre>` styling block (dark bg, monospace, rounded)
- Cards use same `card p-3` pattern as existing "What is Kriging?" cards
- No images needed for this section
- No em-dashes in new copy

---

## Files Changed

- `docs/index.html` — insert one `<section>` block

