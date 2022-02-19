#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os


def _get_base_path(limit=3):
    # Go 'limit' components back in current files's absolute path to arrive at
    # the folder where you can then descend and discover the notebooks folder
    base_path = os.path.abspath(__file__).split(os.path.sep)[: (limit * -1)]
    return base_path


def get_stable_notebooks_folder(logger=None):
    limit = 3
    base_path = _get_base_path(limit=limit)
    stable_notebooks_folder = os.path.sep.join(base_path + ["notebooks", "stable"])

    if logger is not None:
        logger.debug("stable notebooks folder is : {0}".format(stable_notebooks_folder))

    return stable_notebooks_folder


def get_datasets_folder(logger=None):
    limit = 3
    base_path = _get_base_path(limit=limit)
    datasets_folder = os.path.sep.join(base_path + ["notebooks", "datasets"])

    if logger is not None:
        logger.debug("datasets folder is : {0}".format(datasets_folder))

    return datasets_folder
