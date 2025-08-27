"""
Setup script for TargetPanelBench.
"""
from setuptools import setup, find_packages
import pathlib

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name="targetpanelbench",
    version="1.0.0",
    description="A benchmark for target prioritization and panel design",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/ArchipelagoAnalytics/TargetPanelBench",
    author="Archipelago Analytics",
    author_email="benchmark@archipelagoanalytics.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Intended Audience :: Science/Research",
        "Development Status :: 4 - Beta",
    ],
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0", 
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "requests>=2.25.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "networkx>=2.6.0",
        "tqdm>=4.60.0"
    ],
    extras_require={
        "cma": ["cma>=3.1.0"],
        "plotting": ["plotly>=5.0.0"],
        "jupyter": ["jupyter>=1.0.0", "notebook>=6.4.0"],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0", 
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "targetpanelbench=scripts.run_benchmark:main",
            "download-data=data.download_data:main"
        ]
    }
)
