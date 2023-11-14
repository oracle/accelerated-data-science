#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import base64
import json
import os
from typing import Dict
from typing import Tuple

import inflection
import yaml
import glob

from ads.common.auth import create_signer
from ads.opctl import logger
from ads.opctl.config.base import ConfigProcessor
from ads.opctl.config.utils import NotSupportedError, convert_notebook
from ads.opctl.constants import (
    ML_JOB_GPU_IMAGE,
    ML_JOB_IMAGE,
)
from ads.opctl.utils import (
    list_ads_operators,
    parse_conda_uri,
    get_region_key,
    get_namespace,
)


YAML_STRUCTURES = {
    "v0": {
        "infrastructure": ["infrastructure"],
        "image": ["execution", "image"],
    },
    "v1": {
        "infrastructure": ["spec", "Infrastructure", "spec"],
    },
}


class ConfigVersioner(ConfigProcessor):
    """
    ads opctl supports multiple yaml file types, kinds and versions.
    Each has its own slightly different yaml structure.
    This goal of this class is to provide a translation from the generic term for a variable
        to it's specific path in it's yaml.
    """

    def __init__(self, config: Dict = None) -> None:
        super().__init__(config)
        self.ads_operators = list_ads_operators()

    def process(self):
        # this function should be run after merging configs
        # conda pack scenarios --
        # - user runs their own scripts and commands -> source_folder + entrypoint + (command)
        # - user runs ADS operator -> name/YAML

        # docker image scenarios --
        # - user runs their own docker image -> image (to run) + (entrypoint) + (command)
        # - user runs ADS operator -> name/YAML

        # TODO: build this out

        return self
