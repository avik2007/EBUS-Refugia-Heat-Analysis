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
from ebus_core.runner import derive_run_id  # noqa: E402


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aebus", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_val = sub.add_parser("validate", help="parse and schema-check a config")
    p_val.add_argument("config", type=Path)
    p_val.set_defaults(func=cmd_validate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
