#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--opctl",
        action="store_true",
        default=False,
        help="run opctl tests",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "opctl: mark opctl tests")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--opctl"):
        skip_opctl = pytest.mark.skip(reason="need --opctl option to run")
        for item in items:
            if "opctl" in item.keywords:
                item.add_marker(skip_opctl)
