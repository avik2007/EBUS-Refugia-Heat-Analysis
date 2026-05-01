"""
Manifest module for the ArgoEBUS MLOps foundation.

Responsibilities:
- Canonicalize a Pydantic config to a hashable dict (sorted keys, drop
  free-form fields like description).
- Compute a sha256 hash of the canonical dict — the run identity used by
  the collision detector.
- Capture run-time provenance: git SHA, conda env, host, timing.
- Read/write manifest.json files and append to the JSONL run registry.

The manifest is the bridge between a config (what was asked for) and its
outputs (what was produced). Once written, a manifest is immutable.
"""
import hashlib
import json
import platform as _platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel

# Fields excluded from the hash because they don't affect run behavior.
# description is free-form annotation. schema_version IS included because
# schema bumps change interpretation of the config.
_HASH_EXCLUDE = {"description"}


def canonical_config_dict(cfg: BaseModel) -> Dict[str, Any]:
    # Convert a Pydantic config to a deterministic, hashable dict.
    # Uses model_dump(mode='json') so dates become ISO strings and tuples
    # become lists — both round-trippable and stable across Python versions.
    # Recursively sorts all dict keys and strips _HASH_EXCLUDE fields so
    # two configs that are functionally identical always produce the same dict.
    # Input: any Pydantic BaseModel (IngestionConfig or AnalysisConfig)
    # Output: dict with sorted keys, no description, all values JSON-serialisable
    raw = cfg.model_dump(mode="json")
    return _sorted_dict(_strip(raw))


def _strip(obj: Any) -> Any:
    # Recursively remove keys that are in _HASH_EXCLUDE from any nested dict.
    # Input: any Python value (dict, list, or scalar)
    # Output: same structure with excluded keys removed from all dicts
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in _HASH_EXCLUDE}
    if isinstance(obj, list):
        return [_strip(x) for x in obj]
    return obj


def _sorted_dict(obj: Any) -> Any:
    # Recursively sort all dict keys to ensure deterministic serialisation.
    # Without this, dict insertion order would make identical configs hash differently
    # on different Python versions or after field reordering.
    # Input: any Python value (dict, list, or scalar)
    # Output: same value with all nested dicts sorted by key
    if isinstance(obj, dict):
        return {k: _sorted_dict(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, list):
        return [_sorted_dict(x) for x in obj]
    return obj


def config_hash(cfg: BaseModel) -> str:
    # Return the sha256 hex digest (64 chars) of the canonicalized config.
    # Same config (modulo description) → same hash on any machine.
    # Different config → different hash with overwhelming probability (2^-256 collision).
    # Input: any Pydantic BaseModel (IngestionConfig or AnalysisConfig)
    # Output: 64-character lowercase hex string (sha256 digest)
    canon = canonical_config_dict(cfg)
    blob = json.dumps(canon, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# Whitelisted package names inlined into every manifest's key_packages dict.
# §A.4: scipy added to support TEOS-10 / gsw calculations.
# Bumping this list requires a schema_version increment; full conda_list.txt
# is the authoritative source for the complete environment snapshot.
KEY_PACKAGES = (
    "scikit-learn", "xarray", "numpy", "pandas", "gsw",
    "coiled", "dask", "matplotlib", "cartopy", "scipy",
)


def _git(args: list, cwd: Path) -> str:
    # Run a git subcommand and return its stripped stdout.
    # Input: args — list of git arguments (e.g. ["rev-parse", "HEAD"])
    # Input: cwd — directory to run git from (must be inside the repo)
    # Output: stripped stdout string from the git command
    # Raises: subprocess.CalledProcessError if git exits non-zero
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True,
    ).stdout.strip()


def _find_repo_root(start: Path) -> Path:
    # Walk upward from start until a .git/ directory is found.
    # Input: start — any path inside the git repo
    # Output: the repo root Path
    # Raises: RuntimeError if no .git found in any parent
    p = start.resolve()
    while p != p.parent:
        if (p / ".git").exists():
            return p
        p = p.parent
    raise RuntimeError(f"no .git directory found in any parent of {start}")


def capture_code(cwd: Optional[Path] = None) -> Dict[str, Any]:
    # Snapshot the source-code state at run time: git SHA, dirty flag, branch, repo root.
    # Input: cwd — optional path to start repo-root search from; defaults to cwd()
    # Output: dict with keys git_sha (40-char hex), git_dirty (bool), git_branch (str),
    #         repo_root (str path)
    # Raises: RuntimeError if not inside a git repo; subprocess.CalledProcessError on git error
    cwd = Path(cwd) if cwd else Path.cwd()
    root = _find_repo_root(cwd)
    sha = _git(["rev-parse", "HEAD"], root)
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], root)
    dirty = bool(_git(["status", "--porcelain"], root))
    return {
        "git_sha": sha,
        "git_dirty": dirty,
        "git_branch": branch,
        "repo_root": str(root),
    }


def capture_env(
    conda_env_name: str,
    conda_list_dest: Optional[Path],
) -> Dict[str, Any]:
    # Snapshot the Python execution environment for reproducibility.
    # Always inlines the KEY_PACKAGES dict (versions or "MISSING" sentinel).
    # If conda_list_dest is given, writes the full conda list --json output there
    # so the manifest can reference it for exact env reconstruction.
    # Input: conda_env_name — name of the active conda environment (e.g. "ebus-cloud-env")
    # Input: conda_list_dest — optional path to write full conda list JSON to; None = skip
    # Output: dict with conda_env_name, python_version, key_packages dict, and
    #         optionally conda_list_full (path to the written file)
    # Raises: subprocess.CalledProcessError if conda list fails
    full = subprocess.run(
        ["conda", "list", "--json", "-n", conda_env_name],
        capture_output=True, text=True, check=True,
    ).stdout
    parsed = json.loads(full)
    versions = {pkg["name"]: pkg["version"] for pkg in parsed}
    key_pkgs = {name: versions.get(name, "MISSING") for name in KEY_PACKAGES}

    out: Dict[str, Any] = {
        "conda_env_name": conda_env_name,
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "key_packages": key_pkgs,
    }
    if conda_list_dest is not None:
        conda_list_dest = Path(conda_list_dest)
        conda_list_dest.parent.mkdir(parents=True, exist_ok=True)
        conda_list_dest.write_text(full)
        out["conda_list_full"] = str(conda_list_dest)
    return out


def capture_host() -> Dict[str, Any]:
    # Snapshot the host machine for audit trail.
    # Input: none
    # Output: dict with hostname (str) and platform (str, e.g. "Linux-6.6.87...")
    return {
        "hostname": socket.gethostname(),
        "platform": _platform.platform(),
    }
