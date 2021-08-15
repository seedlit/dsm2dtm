import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dsm2dtm",
    version="0.1.0",
    author="Naman Jain",
    author_email="naman.jain@btech2015.iitgn.ac.in",
    description="Generate DTM (Digital Terrain Model) from DSM (Digital Surface Model)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seedlit/dsm2dtm",
    py_modules=['dsm2dtm'], # Sol for single file package: https://docs.python.org/3/distutils/introduction.html#a-simple-example
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Development Status :: 4 - Beta",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.20.3",
        "GDAL>=3.0.4",
        "rasterio>=1.2.5",        
    ],
)