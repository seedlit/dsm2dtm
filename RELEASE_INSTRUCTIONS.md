# Release Instructions for dsm2dtm

This project uses automated workflows for publishing to PyPI and Conda-Forge.

## How to Release a New Version

1.  **Update Version:**
    *   Edit `pyproject.toml` and bump the version (e.g., from `0.3.0` to `0.4.0`).
    *   *(Optional but recommended)* Update `CHANGELOG.md` or release notes.

2.  **Commit and Push:**
    ```bash
    git add pyproject.toml
    git commit -m "Bump version to 0.4.0"
    git push origin main
    ```

3.  **Draft a Release on GitHub:**
    *   Go to the repository on GitHub.
    *   Click **Releases** > **Draft a new release**.
    *   **Tag:** Create a new tag matching the version, e.g., `v0.4.0`.
    *   **Target:** `main`.
    *   **Description:** Add your release notes.
    *   Click **Publish release**.

4.  **Automated Actions:**
    *   **PyPI:** The GitHub release triggers the `.github/workflows/pypi-publish.yml` workflow, which builds and uploads the package to [PyPI](https://pypi.org/project/dsm2dtm/).
    *   **Conda-Forge:** The `regro-cf-autotick-bot` detects the new version on PyPI and automatically opens a Pull Request on the [dsm2dtm-feedstock](https://github.com/conda-forge/dsm2dtm-feedstock) repository.

5.  **Final Step (Conda):**
    *   Go to the `dsm2dtm-feedstock` PR.
    *   Verify the checkmarks (green builds).
    *   Merge the PR.
    *   The package will be available on `conda-forge` shortly.

## Testing Locally (Optional)

To verify the build works locally before releasing:

```bash
uv run python -m build
twine check dist/*
```
