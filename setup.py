from setuptools import setup, find_packages
from pathlib import Path

# see https://packaging.python.org/guides/single-sourcing-package-version/
version_dict = {}
with open(Path(__file__).parents[0] / "herobm/_version.py") as fp:
    exec(fp.read(), version_dict)
version = version_dict["__version__"]
del version_dict

setup(
    name="herobm",
    version=version,
    author="Daniele Angioletti, ---",
    description="HEroBM is a tool for back-mapping coarse-grained simulations to atomistic resolution.",
    python_requires=">=3.8",
    packages=find_packages(include=["herobm", "herobm.*"]),
    entry_points={
        # make the scripts available as command line scripts
        "console_scripts": [
            "herobm-dataset = herobm.scripts.build_dataset:main",
            "herobm-backmap = herobm.scripts.run_inference:main",
        ]
    },
    install_requires=[
        "ipykernel",
        "matplotlib",
        "MDAnalysis",
        "pandas",
        "plotly",
        "nbformat",
    ],
    zip_safe=True,
)