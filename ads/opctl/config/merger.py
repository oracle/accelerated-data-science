#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from string import Template
from typing import Dict
import json

import yaml

from ads.common.auth import AuthType
from ads.opctl import logger
from ads.opctl.config.base import ConfigProcessor
from ads.opctl.config.utils import read_from_ini, _DefaultNoneDict
from ads.opctl.utils import is_in_notebook_session, get_service_pack_prefix
from ads.opctl.constants import (
    DEFAULT_PROFILE,
    DEFAULT_OCI_CONFIG_FILE,
    DEFAULT_CONDA_PACK_FOLDER,
    DEFAULT_ADS_CONFIG_FOLDER,
    ADS_JOBS_CONFIG_FILE_NAME,
    ADS_CONFIG_FILE_NAME,
    ADS_JOBS_CONFIG_FILE_NAME,
    ADS_DATAFLOW_CONFIG_FILE_NAME,
    DEFAULT_NOTEBOOK_SESSION_CONDA_DIR,
)


class ConfigMerger(ConfigProcessor):
    """Merge configurations from command line args, YAML, ini and default configs.
    The order of precedence is
    command line args > YAML + conf.ini > .ads_ops/config.ini + .ads_ops/ml_job_config.ini or .ads_ops/dataflow_config.ini
    Detailed examples can be found at the last section on this page:
    https://bitbucket.oci.oraclecorp.com/projects/ODSC/repos/advanced-ds/browse/ads/opctl/README.md?at=refs%2Fheads%2Fads-ops-draft
    """

    def process(self, **kwargs) -> None:
        config_string = Template(json.dumps(self.config)).substitute(os.environ)
        self.config = json.loads(config_string)
        # 1. merge and overwrite values from command line args
        self._merge_config_with_cmd_args(kwargs)
        # 1.5 merge environment variables
        # TODO

        # 2. fill in values from conf file
        self._fill_config_from_conf()

        ads_config_path = os.path.abspath(
            os.path.expanduser(
                self.config["execution"].pop("ads_config", DEFAULT_ADS_CONFIG_FOLDER)
            )
        )
        # 3. fill in values from default files under ~/.ads_ops
        self._fill_config_with_defaults(ads_config_path)

        logger.debug(f"Config: {self.config}")
        return self

    def _merge_config_with_cmd_args(self, cmd_args: Dict) -> None:
        # overwrite config with command line args
        # if a command line arg value is None or empty collection, then it is ignored
        def _overwrite(cfg, args):
            for k, v in cfg.items():
                if isinstance(v, dict):
                    _overwrite(v, args)
                elif k in args:
                    if (
                        isinstance(args[k], bool)
                        or (isinstance(args[k], list) and len(args[k]) > 0)
                        or args[k]
                    ):
                        cfg[k] = args.pop(k)
                    else:
                        args.pop(k)

        _overwrite(self.config, cmd_args)

        # save everything else from command line in "execution" section
        if "execution" not in self.config:
            self.config["execution"] = {}
        for k, v in cmd_args.items():
            if isinstance(v, bool) or (isinstance(v, list) and len(v) > 0) or v:
                self.config["execution"][k] = v

    def _fill_config_from_conf(self) -> None:
        if self.config["execution"].get("conf_file"):
            conf_file = self.config["execution"]["conf_file"]
            conf_profile = self.config["execution"].get("conf_profile", DEFAULT_PROFILE)
            logger.info(f"Reading from {conf_profile} using profile {conf_profile}")
            parser = read_from_ini(conf_file)
            extra_configs = dict(os.environ)
            extra_configs.update(parser[conf_profile])
            config_string = yaml.dump(self.config)
            # _DefaultNoneDict is used so that if $variable is not found in a section in .ini, None value is filled in.
            self.config = yaml.safe_load(
                Template(config_string).substitute(_DefaultNoneDict(**extra_configs))
            )

    def _fill_config_with_defaults(self, ads_config_path: str) -> None:
        exec_config = self._get_default_config()
        exec_config_from_conf = self._get_config_from_config_ini(ads_config_path)
        exec_config.update(
            {k: v for k, v in exec_config_from_conf.items() if v is not None}
        )
        # set default auth
        if not self.config["execution"].get("auth", None):
            if is_in_notebook_session():
                self.config["execution"]["auth"] = AuthType.RESOURCE_PRINCIPAL
            else:
                self.config["execution"]["auth"] = AuthType.API_KEY
        # determine profile
        if self.config["execution"]["auth"] != AuthType.API_KEY:
            profile = self.config["execution"]["auth"].upper()
            exec_config.pop("oci_profile", None)
            self.config["execution"]["oci_profile"] = None
        else:
            profile = self.config["execution"].get("oci_profile") or exec_config.get(
                "oci_profile"
            )
            self.config["execution"]["oci_profile"] = profile
        # loading config for corresponding profile
        logger.info(f"Loading service config for profile {profile}.")
        infra_config = self._get_service_config(profile, ads_config_path)
        if infra_config.get(
            "conda_pack_os_prefix"
        ):  # this is a field that appeared both in config.ini and ml_job_config.ini
            exec_config["conda_pack_os_prefix"] = infra_config.pop(
                "conda_pack_os_prefix"
            )
        for k, v in exec_config.items():
            if v and not self.config["execution"].get(k):
                self.config["execution"][k] = v
        if not self.config.get("infrastructure"):
            self.config["infrastructure"] = {}
        for k, v in infra_config.items():
            if v and not self.config["infrastructure"].get(k):
                self.config["infrastructure"][k] = v

    def _get_default_config(self) -> Dict:
        if is_in_notebook_session():
            conda_pack_os_prefix = get_service_pack_prefix()
            return {
                "oci_config": DEFAULT_OCI_CONFIG_FILE,
                "oci_profile": DEFAULT_PROFILE,
                "conda_pack_folder": DEFAULT_NOTEBOOK_SESSION_CONDA_DIR,
                "conda_pack_os_prefix": conda_pack_os_prefix,
            }
        else:
            return {
                "oci_config": DEFAULT_OCI_CONFIG_FILE,
                "oci_profile": DEFAULT_PROFILE,
                "conda_pack_folder": DEFAULT_CONDA_PACK_FOLDER,
            }

    @staticmethod
    def _get_config_from_config_ini(ads_config_folder: str) -> Dict:
        if os.path.exists(os.path.join(ads_config_folder, ADS_CONFIG_FILE_NAME)):
            parser = read_from_ini(
                os.path.join(ads_config_folder, ADS_CONFIG_FILE_NAME)
            )
            return {
                "oci_config": parser["OCI"].get("oci_config"),
                "oci_profile": parser["OCI"].get("oci_profile"),
                "conda_pack_folder": parser["CONDA"].get("conda_pack_folder"),
                "conda_pack_os_prefix": parser["CONDA"].get("conda_pack_os_prefix"),
            }
        else:
            logger.info(
                f"{os.path.join(ads_config_folder, 'config.ini')} does not exist. No config loaded."
            )
            return {}

    def _get_service_config(self, oci_profile: str, ads_config_folder: str) -> Dict:
        if self.config["execution"].get("backend", None) == "dataflow":
            config_file = ADS_DATAFLOW_CONFIG_FILE_NAME
        else:
            config_file = ADS_JOBS_CONFIG_FILE_NAME

        if os.path.exists(os.path.join(ads_config_folder, config_file)):
            parser = read_from_ini(os.path.join(ads_config_folder, config_file))
            if oci_profile in parser:
                return parser[oci_profile]
        else:
            logger.info(
                f"{os.path.join(ads_config_folder, config_file)} does not exist. No config loaded."
            )
        return {}
