#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import sys
import os
import json
from setuptools import setup, find_packages
from functools import reduce
from pathlib import Path
from setuptools.command.install import install
from setuptools.command.develop import develop


install_requires = [
    "asteval>=0.9.25",
    "cerberus>=1.3.4",
    "cloudpickle>=1.6.0",
    "fsspec>=0.8.7",
    "jinja2>=2.11.2",
    "gitpython>=3.1.2",
    "matplotlib>=3.1.3",
    "numpy>=1.19.2",
    "oci>=2.104.3",
    "ocifs>=1.1.3",
    "pandas>1.2.1,<1.6",
    "python_jsonschema_objects>=0.3.13",
    "PyYAML>=6",  # pyyaml 5.4 is broken with cython 3
    "requests",
    "scikit-learn>=0.23.2,<1.2",
    "tabulate>=0.8.9",
    "tqdm>=4.59.0",
    "psutil>=5.7.2",
]

extras_require = {
    "boosted": [
        "xgboost",
        "lightgbm",
    ],
    "notebook": [
        "ipywidgets~=7.6.3",
        "ipython>=7.23.1, <8.0",
    ],
    "text": ["wordcloud>=1.8.1", "spacy"],
    "viz": [
        "bokeh>=2.3.0, <=2.4.3",
        "folium>=0.12.1",
        "graphviz<0.17",
        "scipy>=1.5.4",
        "seaborn>=0.11.0",
    ],
    "data": [
        "fastavro>=0.24.2",
        "openpyxl>=3.0.7",
        "pandavro>=1.6.0",
        "datefinder>=0.7.1",
        "htmllistparse>=0.6.0",
        "sqlalchemy>=1.4.1, <=1.4.46",
        "oracledb>=1.0",
    ],
    "opctl": [
        "oci-cli",
        "docker",
        "conda-pack",
        "nbconvert",
        "nbformat",
        "inflection",
    ],
    "bds": ["ibis-framework[impala]", "hdfs[kerberos]", "sqlalchemy"],
    "spark": ["pyspark>=3.0.0"],
    "huggingface": ["transformers"],
}

this_directory = Path(__file__).parent


def update_extra_with_internal_packages():
    loaded_dep = {}
    internal_deps = os.path.join(this_directory, "internal_extra_dependency.json")
    print(f"looking for {internal_deps}")
    if os.path.exists(internal_deps):
        with open(internal_deps) as idf:
            loaded_dep = json.load(idf)
            print(f"Found: {loaded_dep}")
    return loaded_dep


extras_require.update(update_extra_with_internal_packages())

extras_require["torch"] = extras_require["viz"] + ["torch"] + ["torchvision"]
extras_require["tensorflow"] = extras_require["viz"] + [
    "tensorflow",
]
extras_require["geo"] = extras_require["viz"] + ["geopandas"]
extras_require["onnx"] = extras_require["viz"] + [
    "protobuf<=3.20",
    "onnx>=1.12.0",
    "onnxruntime>=1.10.0",
    "onnxmltools>=1.10.0",
    "skl2onnx>=1.10.4",
    "tf2onnx",
    "xgboost==1.5.1",
    "lightgbm==3.3.1",
]
extras_require["optuna"] = extras_require["viz"] + ["optuna==2.9.0"]

extras_require["complete"] = sorted({v for req in extras_require.values() for v in req})
extras_require["all-optional"] = reduce(
    list.__add__,
    [
        extras_require[k]
        for k in extras_require
        if k not in ["boosted", "opctl", "complete"]
    ],
)
extras_require["all-public"] = reduce(
    list.__add__,
    [
        extras_require[k]
        for k in extras_require
        if k not in ["all-optional", "complete"]
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


long_description = (this_directory / "README.md").read_text()
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
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="Oracle Cloud Infrastructure, OCI, Machine Learning, ML, Artificial Intelligence, AI, Data Science, Cloud, Oracle",
    include_package_data=True,
    install_requires=install_requires,
    python_requires=">=3.8",
    setup_requires=setup_requires,
    extras_require=extras_require,
    tests_require=[
        "pytest",
    ],
    project_urls={
        "Github": "https://github.com/oracle/accelerated-data-science",
        "Documentation": "https://accelerated-data-science.readthedocs.io/en/latest/index.html",
    },
    entry_points={"console_scripts": ["ads=ads.cli:cli"]},
)
