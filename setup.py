#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import re
from os import path

from setuptools import find_packages, setup

PACKAGE_NAME = "text2phonemefast"
here = path.abspath(path.dirname(__file__))

with io.open("%s/__init__.py" % PACKAGE_NAME, "rt", encoding="utf8") as f:
    version = re.search(r"__version__ = \"(.*?)\"", f.read()).group(1)

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3",
]

setup(
    name="text2phonemefast",
    version=version,
    description="A Python Library to convert text to phoneme sequence fast - used for XPhoneBERT (Forked and enhanced from original work by Linh The Nguyen)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/manhcuong02/Text2PhonemeFast',
    author="Nguyễn Mạnh Cường",
    author_email="manhcuong17072002@gmail.com",
    maintainer="Nguyễn Mạnh Cường",
    maintainer_email="manhcuong17072002@gmail.com",
    # Original author: Linh The Nguyen (toank45sphn@gmail.com)
    classifiers=classifiers,
    keyword="text2phonemefast",
    packages=find_packages(),
    install_requires=open(path.join(here, 'requirements.txt'), encoding='utf-8').read().splitlines(),
    python_requires=">=3.10",
)
