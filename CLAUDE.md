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

6. **Don't add verbose comments.** Default to no comments. Trust well-named identifiers and docstrings. Only add a comment when the WHY is non-obvious (a hidden constraint, a workaround, surprising behavior). Never restate WHAT the code does (`# Get parameters`, `# close source`, `# Read raster data`). Never write multi-line block comments to explain a section — if a section needs that much explanation, it probably needs to be a function with a one-line docstring.

7. **Use the right tools to verify currency.** Before writing API code (especially against external libraries — rasterio, GDAL, scipy, QGIS Processing, GitHub Actions, etc.), reach for the resources that confirm we're using current, correct usage:
   - `context7` MCP — fetch current docs for the specific library/version we depend on; preferred over recall for any library API.
   - Web search / `WebFetch` — release notes, deprecation warnings, GitHub issues, recent Stack Overflow answers.
   - Available skills (`Skill` tool) — invoke any that fit the task (debugging, brainstorming, frontend-design, etc.).
   - Plugins / subagents (`Agent`, `Explore`) — for cross-cutting research and parallel investigation.
   Cite the source when a non-obvious choice is driven by it.

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

### Live testing the QGIS plugin (mandatory for plugin changes)

QGIS is installed locally at `/Applications/QGIS.app`. Any change under `qgis_plugin/` MUST be exercised through real QGIS — proxy testing via rasterio is not enough (different code paths, no actual `osgeo` import, no QGIS feedback object).

Setup (one-time per machine):
1. Symlink the dev plugin into the QGIS profile:
   ```
   ln -s "$(pwd)/qgis_plugin/dsm2dtm" "$HOME/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/dsm2dtm"
   ```
   (Move any existing `dsm2dtm` install aside first.)
2. Stage test data under `/tmp/` — `qgis_process` cannot read `~/Library/Caches/` on macOS due to sandboxing.

Per-change verification:
```
/Applications/QGIS.app/Contents/MacOS/bin/qgis_process run dsm2dtm:dsm_to_dtm \
  --INPUT=/tmp/dsm2dtm_livetest/<file>.tif \
  --RADIUS=40 --SLOPE=0 \
  --OUTPUT=/tmp/dsm2dtm_livetest/dtm_out.tif
```

Test BOTH a projected CRS input AND a geographic CRS input (re-warp one of the release fixtures with `gdalwarp -t_srs EPSG:4326`). Confirm:
- the run completes,
- the geographic case logs `Input is geographic ... Reprojecting to EPSG:...`,
- output preserves the input CRS, shape, and bounds,
- a meaningful fraction of cells had height removed (`DSM - DTM > 5m` for many cells).
