# Argo Project — Technical Spec Probe
## Purpose
Extract exact technical specs from ArgoEBUSAnalysis for Wherobots interview prep.
Run this prompt in `/home/avik2007/ArgoEBUSAnalysis` with Claude.

---

## Prompt to run

"I have a job interview at Wherobots for a Senior ML Engineer, GeoAI Platform role. They care about:
- Distributed data pipelines (Dask, Zarr, Xarray)
- I/O and memory performance at scale
- Geospatial raster data handling (chunking, regridding, spatial alignment)
- Research-to-production discipline

Please explore this codebase and extract exact, specific answers to the following questions. Quote file paths and line numbers where relevant. Do NOT summarize generically — I need exact specs I can cite in an interview.

1. What distributed compute framework is used (Dask, SLURM, both)? Where in the code?
2. What is the total data volume processed? What formats (NetCDF, Zarr, HDF5)?
3. How is chunking handled — what chunk sizes, along which dimensions (time, lat, lon, depth)?
4. How are multiple data sources aligned spatially (regridding, interpolation method, resolution)?
5. What was the I/O bottleneck and how was it addressed (if at all)?
6. What ML model is trained on the output? What framework? Input/output tensor shapes?
7. What does 'production-grade' look like here — is there a pipeline that runs end-to-end reproducibly, or is it exploratory notebooks?
8. Any SLURM job scripts — what resource requests (nodes, memory, walltime)?
9. What is the final measurable outcome (accuracy metric, benchmark improvement, etc.)?
10. Any Zarr v2/v3 usage specifically — stores, chunk specs, compressors used?"
