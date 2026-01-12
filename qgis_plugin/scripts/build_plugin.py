#!/usr/bin/env python3
"""Build script for QGIS plugin packaging.

This script:
1. Cleans previous build artifacts
2. Vendors the dsm2dtm library into the plugin
3. Creates a distributable ZIP file

Usage:
    python qgis_plugin/scripts/build_plugin.py
"""

import shutil
import zipfile
from pathlib import Path

# Paths
QGIS_PLUGIN_ROOT = Path(__file__).parent.parent  # qgis_plugin/
PROJECT_ROOT = QGIS_PLUGIN_ROOT.parent  # dsm2dtm/
SRC_LIB = PROJECT_ROOT / "src" / "dsm2dtm"
PLUGIN_DIR = QGIS_PLUGIN_ROOT / "dsm2dtm"
EXT_LIBS = PLUGIN_DIR / "ext_libs"
OUTPUT_DIR = QGIS_PLUGIN_ROOT / "dist"
OUTPUT_ZIP = OUTPUT_DIR / "dsm2dtm_qgis_plugin.zip"


def clean():
    """Remove previous build artifacts."""
    if EXT_LIBS.exists():
        shutil.rmtree(EXT_LIBS)
        print(f"✓ Cleaned {EXT_LIBS}")
    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()
        print(f"✓ Removed old {OUTPUT_ZIP}")


def vendor_library():
    """Copy dsm2dtm source into ext_libs for vendoring.

    Note: We rename to 'dsm2dtm_core' to avoid naming conflict with the plugin package.
    """
    dest = EXT_LIBS / "dsm2dtm_core"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SRC_LIB, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))

    # Patch internal imports: dsm2dtm -> dsm2dtm_core
    for py_file in dest.glob("*.py"):
        content = py_file.read_text()
        patched = content.replace("from dsm2dtm.", "from dsm2dtm_core.")
        patched = patched.replace("import dsm2dtm.", "import dsm2dtm_core.")
        py_file.write_text(patched)

    print(f"✓ Vendored dsm2dtm as dsm2dtm_core to {dest}")


def create_zip():
    """Create distributable ZIP file for QGIS plugin manager."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add plugin files
        for file in PLUGIN_DIR.rglob("*"):
            if file.is_file() and "__pycache__" not in str(file):
                # Archive path should be dsm2dtm/... (plugin folder name)
                arcname = file.relative_to(QGIS_PLUGIN_ROOT)
                zf.write(file, arcname)

        # Add LICENSE file from project root
        license_file = PROJECT_ROOT / "LICENSE"
        if license_file.exists():
            # Place LICENSE inside the dsm2dtm folder in the zip
            zf.write(license_file, "dsm2dtm/LICENSE")
            print("✓ Added LICENSE")
        else:
            print("! Warning: LICENSE file not found in project root")

    print(f"✓ Created {OUTPUT_ZIP}")

    # Print zip contents for verification
    print("\nZIP contents:")
    with zipfile.ZipFile(OUTPUT_ZIP, "r") as zf:
        for name in sorted(zf.namelist())[:20]:
            print(f"  {name}")
        if len(zf.namelist()) > 20:
            print(f"  ... and {len(zf.namelist()) - 20} more files")


def main():
    """Main build entry point."""
    print("=" * 50)
    print("Building QGIS Plugin for dsm2dtm")
    print("=" * 50)
    print()

    # Validate paths
    if not SRC_LIB.exists():
        print(f"✗ Error: Source library not found at {SRC_LIB}")
        return 1
    if not PLUGIN_DIR.exists():
        print(f"✗ Error: Plugin directory not found at {PLUGIN_DIR}")
        return 1

    clean()
    vendor_library()
    create_zip()

    print()
    print("=" * 50)
    print("Build complete!")
    print("=" * 50)
    print()
    print("To install:")
    print("1. Open QGIS")
    print("2. Go to: Plugins → Manage and Install Plugins → Install from ZIP")
    print(f"3. Select: {OUTPUT_ZIP}")
    print()

    return 0


if __name__ == "__main__":
    exit(main())
