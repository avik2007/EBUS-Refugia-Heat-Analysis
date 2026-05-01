"""
aebus — config-driven CLI for the ArgoEBUS pipeline.

Subcommands:
  validate  Parse + schema-check a config and print its derived run_id.
  ingest    Run an ingestion config end-to-end (writes parquet + manifest).
  analyze   Run an analysis config end-to-end (writes audit + manifest).
  list      Tabulate runs from AEResults/run_registry.jsonl.
  show      Print a run's manifest.json.

All commands invoke library code under ebus_core.{config_schema, manifest, runner}.
"""
import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

# Allow `python ArgoEBUSCloud/aebus_cli.py ...` from the repo root by
# pointing imports at the ArgoEBUSCloud directory.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from ebus_core.config_schema import load_config  # noqa: E402
from ebus_core.manifest import ManifestCollisionError  # noqa: E402
from ebus_core.runner import derive_run_id, run_analysis, run_ingestion  # noqa: E402


DEFAULT_REGISTRY = Path("AEResults/run_registry.jsonl")


def cmd_validate(args: argparse.Namespace) -> int:
    """Parse + schema-check config; print run_id and target manifest path."""
    try:
        cfg = load_config(args.config)
    except (ValidationError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    rid = derive_run_id(cfg)
    print(f"config_kind: {cfg.config_kind}")
    print(f"run_id:      {rid}")
    if cfg.config_kind == "analysis":
        print(f"manifest:    AEResults/aelogs/{rid}/manifest.json")
    else:
        print(f"manifest:    AEResults/aelogs/ingestion/{rid}/manifest.json")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Load an analysis config and execute the full GPR pipeline."""
    try:
        cfg = load_config(args.config)
    except (ValidationError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    if cfg.config_kind != "analysis":
        print(f"ERROR: expected analysis config, got {cfg.config_kind}", file=sys.stderr)
        return 2
    try:
        result = run_analysis(
            cfg,
            registry_path=args.registry,
            force_overwrite=args.force_overwrite,
        )
    except ManifestCollisionError as e:
        print(f"COLLISION: {e}", file=sys.stderr)
        return 3
    print(json.dumps(result, indent=2))
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    """Load an ingestion config and execute the full cloud-ingest pipeline."""
    try:
        cfg = load_config(args.config)
    except (ValidationError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    if cfg.config_kind != "ingestion":
        print(f"ERROR: expected ingestion config, got {cfg.config_kind}", file=sys.stderr)
        return 2
    try:
        result = run_ingestion(
            cfg,
            registry_path=args.registry,
            force_overwrite=args.force_overwrite,
        )
    except ManifestCollisionError as e:
        print(f"COLLISION: {e}", file=sys.stderr)
        return 3
    print(json.dumps(result, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aebus", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_val = sub.add_parser("validate", help="parse and schema-check a config")
    p_val.add_argument("config", type=Path)
    p_val.set_defaults(func=cmd_validate)

    p_ana = sub.add_parser("analyze", help="run an analysis config end-to-end")
    p_ana.add_argument("config", type=Path)
    p_ana.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    p_ana.add_argument("--force-overwrite", action="store_true")
    p_ana.set_defaults(func=cmd_analyze)

    p_ing = sub.add_parser("ingest", help="run an ingestion config end-to-end")
    p_ing.add_argument("config", type=Path)
    p_ing.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    p_ing.add_argument("--force-overwrite", action="store_true")
    p_ing.set_defaults(func=cmd_ingest)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
