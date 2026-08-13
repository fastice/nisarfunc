# CLAUDE.md — nisarfunc

**DEPRECATED.** Superseded by [`nisardev`](../nisardev/CLAUDE.md) (for cal/val
classes) and [`nisarhdf`](../nisarhdf/CLAUDE.md) (for HDF5 reading). Do not add
new code here — this file exists only to help identify old code that still
imports from `nisarfunc`.

## What's in here

| Module | Contents | Replacement |
|---|---|---|
| `nisarBase2D.py` | Abstract base class for polar-stereo 2D image objects (geometry, interpolation, Tiff I/O) | `nisardev.nisarBase2D` |
| `nisarVel.py` | `nisarVel` — single velocity map (vx, vy, v, ex, ey, e); read/write GeoTIFF, `interp`, `displayVel` | `nisardev.nisarVel` / `nisardev.nisarVelSeries` |
| `cvPoints.py` | GPS cal/val point comparison against `nisarVel` | `nisardev.cvPoints` |
| `nisarSupport.py` | Helper functions: `setKey`, `myError`, `readGeoTiff`, `parseDatesFromDirName`, `parseDatesFromMeta` | `nisardev.nisarSupport` |
| `makeCWISPParFromRSLC.py` | `makeCWISPParFromRSLC` — builds a CWISP-format `.par` JSON template from an RSLC HDF5 via `nisarhdf` | no direct replacement; standalone script-style utility |
| `speckleSim.py` | Speckle-tracking simulation utilities (`whiteNoisePatch`, `nccPatches`, `correlatedShiftedPatches`, `speckleJobs`, `osSubPix`, `osSubPixGaussian`, `gaussFit`, etc.) | no direct replacement; standalone simulation code |

## Notes

- `nisarBase2D`/`nisarVel` here are an earlier, less complete version of the
  classes in `nisardev` — e.g. read/write via plain GeoTIFF + GDAL rather than
  `nisardev`'s xarray/rioxarray/dask COG pipeline, and no time-series (`*Series`)
  or image (`nisarImage`) classes.
- `makeCWISPParFromRSLC` and `speckleSim` have no equivalents in `nisardev`; if
  this functionality is needed going forward, treat it as standalone utility code
  rather than as part of a class hierarchy to extend.
- If you encounter old notebooks/scripts importing `nisarfunc`, prefer migrating
  them to `nisardev` rather than fixing bugs in place.
