"""Setup file."""

import os

from setuptools import find_packages, setup

PACKAGE_NAME = "src"
VERSION_FILE_PATH = os.path.join(PACKAGE_NAME, "__init__.py")
README_FILE_PATH = "README.md"


def get_version_from_init() -> str:
    """Retrieve the package version from __init__.py without executing it.

    Returns:
        str: The package version.

    Raises:
        ValueError: If '__version__' is not found in the specified file.
    """
    with open(VERSION_FILE_PATH, encoding="utf-8") as version_file:
        for line in version_file:
            if line.startswith("__version__"):
                # Extract version using string manipulation
                return line.split("=")[-1].strip().strip('"')
    raise ValueError(f"'__version__' not found in '{VERSION_FILE_PATH}'.")


def get_content_from_readme() -> str:
    """Retrieve the content of the README file.

    Returns:
        str: The content of the README file.
    """
    with open(README_FILE_PATH, encoding="utf-8") as readme_file:
        return readme_file.read()


setup(
    name="InferenceService",
    version=get_version_from_init(),
    author="Gordeev A.",
    description="Inference service barcode detection and recognition",
    long_description=get_content_from_readme(),
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    packages=find_packages(exclude=(".github", "docs", "examples")),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.10",
    ],
)
