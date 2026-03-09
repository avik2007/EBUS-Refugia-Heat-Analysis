import numpy as np
import os

def get_ebus_registry():
    """
    Official Spatio-Temporal boundaries for Global EBUS systems.
    """
    return {
        "california": {
            "lat": (30.0, 35.0),
            "lon": (-125.0, -120.0),
            "s3_bucket": "argo-ebus-project-data-abm",
            "hemisphere": "north"
        },
        "humboldt": {
            "lat": (-20.0, -5.0),
            "lon": (-85.0, -70.0),
            "s3_bucket": "argo-ebus-humboldt-abm",
            "hemisphere": "south"
        },
        "canary": {
            "lat": (15.0, 30.0),
            "lon": (-25.0, -13.0),
            "s3_bucket": "argo-ebus-canary-abm",
            "hemisphere": "north"
        },
        "benguela": {
            "lat": (-35.0, -15.0),
            "lon": (5.0, 20.0),
            "s3_bucket": "argo-ebus-benguela-abm",
            "hemisphere": "south"
        }
    }

def get_ae_config(region="california", lat_step=1.0, lon_step=1.0, time_step=30.0):
    """
    Fetch configuration for a target study site with dynamic resolutions.
    """
    registry = get_ebus_registry()
    if region not in registry:
        raise ValueError(f"Region '{region}' not found.")
    
    config = registry[region]
    config["resolutions"] = {
        "lat_step": lat_step,
        "lon_step": lon_step,
        "time_step": time_step
    }
    
    # Unique ID for tracking validation runs
    config["run_id"] = f"{region}_res{lat_step}x{lon_step}_t{time_step}"
    
    # Attach the standardized project paths
    config["paths"] = get_project_paths()
    
    return config

def get_project_paths():
    """
    Calculates paths to AEResults relative to the ArgoEBUSCloud directory.
    Assumes structure: /ArgoEBUSAnalysis/[ArgoEBUSCloud, AEResults]
    """
    # Get the directory where this utils file lives (ebus_core)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels to reach /ArgoEBUSAnalysis/
    project_root = os.path.abspath(os.path.join(base_dir, "..", ".."))
    
    results_dir = os.path.join(project_root, "AEResults")
    
    return {
        "root": project_root,
        "results": results_dir,
        "plots": os.path.join(results_dir, "aeplots"),
        "models": os.path.join(results_dir, "aemodels")
    }

def ensure_ae_dirs():
    """Guarantees the /ArgoEBUSAnalysis/AEResults folders exist."""
    paths = get_project_paths()
    for key in ["plots", "models"]:
        os.makedirs(paths[key], exist_ok=True)

def calculate_bin(value, step):
    """Generic binning helper."""
    return np.floor(value / step) * step