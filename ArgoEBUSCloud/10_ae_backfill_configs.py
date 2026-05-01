"""
Backfill legacy aelogs runs into configs/ schema.

Usage:
    conda run -n ebus-cloud-env python ArgoEBUSCloud/10_ae_backfill_configs.py

Writes one configs/{region}/{run_id}.yaml per aelogs dir.
"""
from pathlib import Path

from ebus_core.backfill import backfill_configs

# Paths relative to project root (script is run from there)
AELOGS_ROOT = Path("AEResults/aelogs")
CONFIGS_ROOT = Path("configs")
TODAY = "2026-04-30"

if __name__ == "__main__":
    written = backfill_configs(AELOGS_ROOT, CONFIGS_ROOT, today=TODAY)
    print(f"Wrote {len(written)} config YAML(s):")
    for p in written:
        print(f"  {p}")
