# Working Agreement (for Claude)

## Workflow Rules

1. **Incremental changes.** Break work into the smallest meaningful steps. One concern per change. Don't bundle unrelated edits.

2. **Verify live before claiming done.** A task is not "done" until the change has been actually executed and the output observed:
   - Run the relevant tests (`uv run pytest <path>`).
   - For non-test code paths, run a focused script or invoke the CLI/plugin directly and check the result.
   - Type-checking and "the code looks right" are NOT verification.
   - If a change cannot be live-tested in this environment (e.g., requires running QGIS GUI), explicitly say so and propose the closest practical verification (e.g., headless invocation, mimicking the data flow in a unit test).

3. **Ask before committing or pushing.** Never run `git commit`, `git push`, or `gh pr create` without explicit approval. Stage changes silently and present a diff summary; wait for "yes, commit" / "yes, push".

4. **Don't auto-commit unrelated files.** `BUG_HUNT_REPORT.md`, `IMPROVEMENT_RECOMMENDATIONS.md`, and `CLAUDE.md` are working notes — don't add them to commits unless asked.

5. **Surface failures honestly.** If a verification step fails or is skipped, say so clearly. No "it should work" claims.

## Project context

- Python library + QGIS plugin for DSM → DTM conversion via Progressive Morphological Filter.
- Two parallel code paths exist today: `src/dsm2dtm/` (rasterio-based, used by CLI) and `qgis_plugin/dsm2dtm/ext_libs/dsm2dtm_core/` (vendored, scipy-only). They have diverged — see BUG-43.
- Test runner: `uv run pytest`. CI: GitHub Actions across 3.11–3.14, ubuntu/macos/windows + conda.
- Bug catalog: `BUG_HUNT_REPORT.md`. Broader recommendations: `IMPROVEMENT_RECOMMENDATIONS.md`.

### Why the library / plugin codebases diverge (intentional)

- **`src/dsm2dtm/` uses rasterio.** The library is published on PyPI and `pip install dsm2dtm` must "just work", so it can rely on rasterio (which ships GDAL in its wheel).
- **`qgis_plugin/` uses GDAL (osgeo) directly.** QGIS does NOT ship rasterio with the standard distribution, but it does ship `osgeo`/GDAL bindings. The plugin must use GDAL primitives so we don't force users to pip-install rasterio into their QGIS Python.
- Implication: any I/O or geospatial transformation in the plugin (read, warp, write) must be expressed via `osgeo.gdal` / `osgeo.osr`, not rasterio. The pure-numpy/scipy algorithm code is shared via the vendored `ext_libs/dsm2dtm_core/`.

### Live test data

For end-to-end testing the same fixtures CI uses are at:
https://github.com/seedlit/dsm2dtm/releases/tag/test-data-v0.1

Includes paired DSM/DTM rasters across resolutions (1m Istanbul hilly urban, 50cm river+urban, 50cm vegetation+urban). Download the ZIP, extract, and feed individual DSMs to the CLI or QGIS plugin to validate end-to-end behavior.
