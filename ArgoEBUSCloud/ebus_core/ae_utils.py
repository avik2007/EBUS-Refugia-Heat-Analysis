import numpy as np
import os

def get_ebus_registry():
    return {
        # THE OLD DEFINITION (For small-scale testing/debugging)
        "california_testbox": {
            "lat": [30.0, 35.0],
            "lon": [-125.0, -120.0],
            "time": ["2015-01-01", "2015-12-31"], # Restoring the time key
            "s3_bucket": "argo-ebus-project-data-abm"
        },
        # THE FULL CCS (For actual scientific analysis)
        "california": {
            "lat": [25.0, 50.0],      
            "lon": [-140.0, -110.0],
            "time": ["2015-01-01", "2015-12-31"], # Restoring the time key
            "s3_bucket": "argo-ebus-project-data-abm"
        },
        "humboldt": {
            "lat": [-45.0, 0.0],     
            "lon": [-90.0, -70.0],
            "time": ["2015-01-01", "2015-12-31"],
            "s3_bucket": "argo-ebus-project-data-abm"
        },
        "canary": {
            "lat": [10.0, 45.0],      
            "lon": [-30.0, -5.0],
            "time": ["2015-01-01", "2015-12-31"],
            "s3_bucket": "argo-ebus-project-data-abm"
        },
        "benguela": {
            "lat": [-35.0, -10.0],    
            "lon": [0.0, 20.0],
            "time": ["2015-01-01", "2015-12-31"],
            "s3_bucket": "argo-ebus-project-data-abm"
        }
    }

def get_ae_config(region="california", lat_step=0.5, lon_step=0.5, time_step=30.0, 
                  depth_range=(0, 100), start_date=None, end_date=None):
    """
    Fetch configuration for a target study site with dynamic resolutions and depth.
    Standardizes naming for S3 files and Coiled clusters.
    """
    registry = get_ebus_registry()
    if region not in registry:
        raise ValueError(f"Region '{region}' not found.")
    
    # Use .copy() to avoid modifying the global registry dictionary
    config = registry[region].copy()
    
    # Dates
    config["start_date"] = start_date if start_date else config["time"][0]
    config["end_date"] = end_date if end_date else config["time"][1]
    
    # Resolutions and Depth
    config["resolutions"] = {
        "lat_step": lat_step,
        "lon_step": lon_step,
        "time_step": time_step
    }
    config["depth_range"] = depth_range
    
    # --- FILENAME GENERATION (run_id) ---
    start_clean = config["start_date"].replace("-", "")
    end_clean = config["end_date"].replace("-", "")
    date_str = f"{start_clean}_{end_clean}"
    
    # Safe naming: periods to underscores for S3/Coiled compatibility
    lat_safe = str(lat_step).replace(".", "_")
    lon_safe = str(lon_step).replace(".", "_")
    time_safe = str(time_step).replace(".", "_")
    depth_str = f"d{depth_range[0]}_{depth_range[1]}"

    # Example: california_20150101_20151231_res0_5x0_5_t30_0_d0_700
    config["run_id"] = f"{region}_{date_str}_res{lat_safe}x{lon_safe}_t{time_safe}_{depth_str}"
    
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
    """Guarantees project directory tree exists."""
    import os
    base_dir = "AEResults"
    for sub in ["aeplots", "aedata", "aelogs"]:
        path = os.path.join(base_dir, sub)
        os.makedirs(path, exist_ok=True)

def calculate_bin(value, step):
    """Generic binning helper."""
    return np.floor(value / step) * step