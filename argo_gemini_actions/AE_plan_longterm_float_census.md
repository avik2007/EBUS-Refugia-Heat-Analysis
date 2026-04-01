# AE Plan: Long-Term Argo Float Census (1999–2025)

## 1. Objective
Identify the optimal spatial and temporal boundaries for the "Stealth Warming" study by mapping historical Argo float density. This will allow us to define `californiav3` based on empirical data hotspots rather than static geographic boxes.

## 2. The Strategy (Plain English)
We need to see how the Argo "network" has breathed over the last 25 years. We will treat the California Current System as a giant checkerboard, where each square is 5 degrees of latitude by 5 degrees of longitude.

For every year between 1999 and 2025, we will:
1.  **Count the Visitors:** Look at every 5x5 degree square and count how many *unique* floats (individual WMO IDs) transmitted data from inside that square during that specific year.
2.  **Map the Density:** Create a heatmap where the "hotter" colors represent more floats.
3.  **Find the Refugia:** By looking at a grid of 26 maps (one for each year), we can visually identify if there is a specific sub-region (like Southern California) that is always "busy" with floats, even in years where the rest of the coast is a "data desert."

This will tell us if we should focus our 3D GP model on a smaller, high-density zone where the Source Layer (150-400m) is actually well-sampled.

## 3. Technical Implementation (Pseudocode)

```python
# SCRIPT: 09_ae_longterm_float_census.py

# STEP 1: LOAD BOUNDS
# Get the original 'california' bounds [25, 50] Lat, [-140, -110] Lon from ae_utils.

# STEP 2: DEFINE THE CHECKERBOARD
# Create bins using np.arange(min, max + 5, 5). 
# This gives us a 5x5 grid across the whole domain.

# STEP 3: DATA ACQUISITION (The Loop)
# Loop through years 1999 to 2025.
# IMPORTANT: Fetch data in 5-year chunks using ebus_core.ae_utils.get_float_history.
# Chunking prevents the ERDDAP server from timing out on a 25-year query.

# STEP 4: SPATIAL BINNING
# For each dive in the dataset:
#   Assign it to a lat_bin: (lat // 5) * 5 + 2.5
#   Assign it to a lon_bin: (lon // 5) * 5 + 2.5

# STEP 5: AGGREGATION
# Group the data by [Year, Lat_Bin, Lon_Bin].
# Use the .nunique() function on 'platform_number' to count UNIQUE floats.
# This is better than counting dives, as one float might dive 30 times in one bin,
# but it still only represents one "sensor" for the GP model.

# STEP 6: VISUALIZATION (Small Multiples)
# Create a large figure (e.g., 20x30 inches).
# Create a grid of subplots (e.g., 6 rows x 5 columns).
# For each year:
#   Draw a map of the California coast using Cartopy.
#   Overlay a heatmap (pcolormesh) of the float counts.
#   Set a fixed color scale (0 to 15+ floats) so years are comparable.

# STEP 7: OUTPUT
# Save the high-resolution PNG to AEResults/aeplots/float_census_california_1999_2025.png.
# Print a table of the top 10 "Data Hotspots" (Year/Bin combinations with most floats).
```

## 4. Expected Outcome
A single "Small Multiples" image that reveals the evolution of the Argo network. From this, the user can say: *"Actually, the Southern California Bight (30-35N) has 10+ floats every year since 2005; let's make californiav3 exactly that size to get a perfect model."*
