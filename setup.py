from setuptools import setup, find_packages

setup(
    name="leible",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "adapters",
        "beautifulsoup4",
        "click",
        "imapclient",
        "joblib",
        "loguru",
        "matplotlib",
        "numpy",
        "openai",
        "polars",
        "python-dotenv",
        "ratelimit",
        "scikit-learn",
        "seaborn",
        "semanticscholar",
        "toolz",
        "torch",
        "transformers",
    ],
    entry_points={
        "console_scripts": [
            "leible=leible.cli:cli",
        ],
    },
)
