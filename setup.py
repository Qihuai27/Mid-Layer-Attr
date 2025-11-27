"""Setup script for icl-anchor-analysis."""

from setuptools import setup, find_packages

setup(
    name="icl-anchor-analysis",
    version="0.1.0",
    description="Analysis of anchor word effects in In-Context Learning",
    author="",
    python_requires=">=3.8",
    packages=find_packages(where=".", include=["src*", "experiments*"]),
    package_dir={"": "."},
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.28.0",
        "datasets>=2.10.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.0.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
    },
)
