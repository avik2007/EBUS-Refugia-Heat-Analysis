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
from typing import Any, Dict

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
