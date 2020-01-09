from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = (
    "torch",
    "numpy",
    "pystiche@https://github.com/pmeier/pystiche/archive/master.zip",
)

classifiers = (
    "Development Status :: 3 - Alpha",
    "License :: OSI Approved :: BSD License",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering",
)

setup(
    name="pystiche_replication",
    description="Replication of prominent NST papers in pystiche",
    version="0.2-dev",
    url="https://github.com/pmeier/pystiche_replication",
    license="BSD-3",
    author="Philip Meier",
    author_email="github.pmeier@posteo.de",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=install_requires,
    python_requires=">=3.6",
    classifiers=classifiers,
)
