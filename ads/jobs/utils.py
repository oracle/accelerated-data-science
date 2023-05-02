#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import os
import json

from oci.data_flow.models import Application

from ads.opctl.config.utils import read_from_ini
from ads.opctl.constants import (
    ADS_DATAFLOW_CONFIG_FILE_NAME,
    DEFAULT_ADS_CONFIG_FOLDER,
)

from ads.jobs import logger
from ads.common.utils import oci_key_profile


def get_dataflow_config(path=None, oci_profile=None):
    if path:
        dataflow_config_file_path = os.path.abspath(os.path.expanduser(path))
    else:
        dataflow_config_file_path = os.path.expanduser(
            os.path.join(DEFAULT_ADS_CONFIG_FOLDER, ADS_DATAFLOW_CONFIG_FILE_NAME)
        )
    config = {}
    if os.path.exists(dataflow_config_file_path):
        parser = read_from_ini(dataflow_config_file_path)
        if not oci_profile:
            oci_profile = oci_key_profile()
        if oci_profile in parser:
            config = dict(parser[oci_profile])
        if len(config) == 0:
            logger.error(
                f"Dataflow configuration with profile {oci_profile} not found."
            )
            raise ValueError(
                f"Dataflow configuration with profile {oci_profile} not found."
            )
        return config
    else:
        logger.warning(f"{dataflow_config_file_path} not found. Follow this link https://accelerated-data-science.readthedocs.io/en/latest/user_guide/apachespark/dataflow.html to set up the config.")
        return {}


class DataFlowConfig(Application):
    def __init__(self, path: str = None, oci_profile: str = None):
        """Create a DataFlowConfig object. If a path to config file is given it is loaded from the path.

        Parameters
        ----------
        path : str, optional
            path to configuration file, by default None
        oci_profile : str, optional
            oci profile to use, by default None
        """
        self.config = get_dataflow_config(path, oci_profile)
        self._script_bucket = None
        self._archive_bucket = None
        if len(self.config) > 0:
            self._script_bucket = self.config.pop("script_bucket")
            self._archive_bucket = self.config.pop("archive_bucket", None)
        super().__init__(**self.config)

    def __repr__(self):
        config = json.loads(super().__repr__())
        config["script_bucket"] = self.script_bucket
        if self.archive_bucket:
            config["archive_bucket"] = self.archive_bucket
        return f"'{json.dumps({k: v for k, v in config.items() if v is not None})}'"

    @property
    def script_bucket(self):
        """Bucket to save user script. Also accept a prefix in the format of oci://<bucket-name>@<namespace>/<prefix>.

        Returns
        -------
        str
            script bucket (path)
        """
        return self._script_bucket

    @script_bucket.setter
    def script_bucket(self, v: str):
        self._script_bucket = v

    @property
    def archive_bucket(self):
        """Bucket to save archive zip. Also accept a prefix in the format of oci://<bucket-name>@<namespace>/<prefix>.

        Returns
        -------
        str :
            archive bucket (path)
        """
        return self._archive_bucket

    @archive_bucket.setter
    def archive_bucket(self, v: str):
        self._archive_bucket = v
