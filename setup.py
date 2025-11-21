"""
Project packaging configuration.

This file is used by setuptools to install the project and its dependencies.
Typical usage:
    pip install -e .
"""

from typing import List

from setuptools import find_packages, setup

#Sometimes used in requirements.txt to trigger editable installs
HYPHEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:
    """
    Read a requirements file and return a list of packages.

    Args:
        file_path: Path to the requirements file (e.g. "requirements.txt").

    Returns:
        A list of requirement strings suitable for `install_requires`.

    Notes:
        - Strips newlines and surrounding whitespace.
        - Ignores empty lines and comment lines starting with "#".
        - Removes the special editable install token "-e ." if present.
    """
    requirements: List[str] = []

    with open(file_path, encoding="utf-8") as file_obj:
        for line in file_obj:
            req = line.strip()

            #Skip empty lines and comments
            if not req or req.startswith("#"):
                continue

            if req == HYPHEN_E_DOT:
                #This is used for editable installs, not an actual dependency
                continue

            requirements.append(req)

    return requirements

setup(
name = "python",
version = "0.0.1",
author = "Godwin",
author_email = "godwinabia@gmail.com",
packages = find_packages(),
install_requires = get_requirements("requirements.txt")
)