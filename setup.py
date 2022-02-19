#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import sys
import os
import json
from setuptools import setup, find_packages
from functools import reduce
from setuptools.command.install import install
from setuptools.command.develop import develop

install_requires = [
    "asteval>=0.9.25",
    "bokeh>=2.3.0",
    "cerberus>=1.3.4",
    "cloudpickle>=1.6.0",
    "cx-Oracle>=8.0",
    "datefinder>=0.7.1",
    "fdk>=0.1.18",
    "folium>=0.12.1",
    "fsspec>=0.8.7",
    "geopandas>=0.9.0",
    "gitpython>=3.1.2",
    "graphviz<0.17",
    "htmllistparse>=0.6.0",
    "ipython>=7.23.1,< 8.0",
    "jinja2>=2.11.2",
    "jsonschema<4.0",
    "matplotlib>=3.1.3",
    "numexpr>=2.7.3",
    "numpy>=1.19.2",
    "oci>=2.48.1",
    "ocifs>=0.1.5",
    "onnx~=1.10.0",
    "onnxmltools~=1.9.0",
    "onnxruntime~=1.8.0",
    "pandas>1.2.1,<1.4",
    "psutil>=5.7.2",
    "py-cpuinfo>=7.0.0",
    "python_jsonschema_objects>=0.3.13",
    "scikit-learn>=0.23.2",
    "scipy>=1.5.4",
    "seaborn>=0.11.0",
    "six>=1.14.0",
    "skl2onnx~=1.9.0",
    "sqlalchemy>=1.4.1",
    "tabulate>=0.8.9",
    "tqdm>=4.59.0",
]

extras_require = {
    "boosted": [
        "xgboost",
        "lightgbm",
    ],
    "notebook": [
        "ipywidgets~=7.6.3",
    ],
    "text": [
        "wordcloud>=1.8.1",
    ],
    "data": [
        "fastavro>=0.24.2",
        "openpyxl>=3.0.7",
        "pandavro>=1.6.0",
    ],
    "opctl": ["oci-cli", "docker"],
    "mysql": ["mysql-connector-python"],
}
extras_require["complete"] = sorted({v for req in extras_require.values() for v in req})
extras_require["all-optional"] = reduce(
    list.__add__,
    [
        extras_require[k]
        for k in extras_require
        if k not in ["labs", "boosted", "opctl"]
    ],
)

# Only include pytest-runner in setup_requires if we're invoking tests
if {"pytest", "test", "ptr"}.intersection(sys.argv):
    setup_requires = ["pytest-runner"]
else:
    setup_requires = []

ADS_VERSION = "UNKNOWN"
with open(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "ads", "ads_version.json")
) as version_file:
    ADS_VERSION = json.load(version_file)["version"]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


class CustomCommandMixin:
    user_options = [("enable-cli", None, "flag to install ADS cli")]

    def initialize_options(self):
        super().initialize_options()
        self.enable_cli = None

    def run(self):
        if self.enable_cli:
            self.distribution.scripts = ["ads/ads"]
        super().run()


class InstallCommand(CustomCommandMixin, install):
    user_options = install.user_options + CustomCommandMixin.user_options


class DevelopCommand(CustomCommandMixin, develop):
    user_options = develop.user_options + CustomCommandMixin.user_options


setup(
    name="oracle_ads",
    version=ADS_VERSION,
    description="Oracle Accelerated Data Science SDK",
    author="Oracle Data Science",
    license="Universal Permissive License 1.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    url="https://docs.oracle.com/en-us/iaas/tools/ads-sdk/latest/index.html",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Universal Permissive License (UPL)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    keywords="Oracle Cloud Infrastructure, OCI, Machine Learning, ML, Artificial Intelligence, AI, Data Science, Cloud, Oracle",
    include_package_data=True,
    install_requires=install_requires,
    python_requires=">=3.7",
    setup_requires=setup_requires,
    extras_require=extras_require,
    tests_require=[
        "pytest",
    ],
    cmdclass={"develop": DevelopCommand, "install": InstallCommand},
)
