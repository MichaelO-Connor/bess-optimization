"""
BESS Optimization Framework
Setup configuration for the Battery Energy Storage System optimization package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
def read_requirements(filename):
    """Read requirements from file."""
    with open(filename, 'r') as f:
        return [line.strip() for line in f 
                if line.strip() and not line.startswith('#')]

# Core requirements
install_requires = read_requirements('requirements.txt')

# Development requirements
extras_require = {
    'dev': read_requirements('requirements-dev.txt'),
    'entsoe': ['entsoe-py>=0.5.10'],
    'viz': ['matplotlib>=3.3.0', 'plotly>=5.0.0'],
}

setup(
    name="bess-optimization",
    version="0.1.0",
    author="BESS Optimization Team",
    author_email="contact@bess-optimization.com",
    description="Battery Energy Storage System optimization framework with MILP solver",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/bess-optimization",
    packages=find_packages(exclude=['tests*', 'examples*', 'docs*']),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        'console_scripts': [
            'bess-optimize=bess.cli:main',
            'bess-validate=bess.cli:validate',
        ],
    },
    include_package_data=True,
    package_data={
        'bess': [
            'data/templates/*.csv',
            'config/*.yaml',
        ],
    },
    zip_safe=False,
    keywords='battery optimization energy storage MILP renewable arbitrage',
    project_urls={
        'Documentation': 'https://bess-optimization.readthedocs.io',
        'Source': 'https://github.com/yourusername/bess-optimization',
        'Tracker': 'https://github.com/yourusername/bess-optimization/issues',
    },
)