from setuptools import setup, find_packages

setup(
    name="leible",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "beautifulsoup4",
        "click",
        "imapclient",
        "joblib",
        "loguru",
        "matplotlib",
        "numpy",
        "polars",
        "python-dotenv",
        "ratelimit",
        "scikit-learn",
        "seaborn",
        "torch",
        "transformers",
    ],
    entry_points={
        "console_scripts": [
            "leible=leible.cli:cli",
        ],
    },
)
