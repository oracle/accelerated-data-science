#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from string import Template
from typing import Dict
import json

import yaml

from ads.opctl import logger
from ads.opctl.config.base import ConfigProcessor
from ads.opctl.config.utils import _read_from_ini, _DefaultNoneDict
from ads.opctl.constants import (
    DEFAULT_PROFILE,
    DEFAULT_OCI_CONFIG_FILE,
    DEFAULT_CONDA_PACK_FOLDER,
    DEFAULT_ADS_CONFIG_FOLDER,
)


class ConfigMerger(ConfigProcessor):
    """Merge configurations from command line args, YAML, ini and default configs.
    The order of precedence is
    command line args > YAML + conf.ini > .ads_ops/config.ini + .ads_ops/ml_job_config.ini
    Detailed examples can be found at the last section on this page:
    https://bitbucket.oci.oraclecorp.com/projects/ODSC/repos/advanced-ds/browse/ads/opctl/README.md?at=refs%2Fheads%2Fads-ops-draft
    """

    def process(self, **kwargs) -> None:
        config_string = Template(json.dumps(self.config)).substitute(os.environ)
        self.config = json.loads(config_string)
        # 1. merge and overwrite values from command line args
        self._merge_config_with_cmd_args(kwargs)
        # 2. fill in values from conf file
        self._fill_config_from_conf()
        # 3. fill in values from default ads ops config, if missing
        ads_config_path = os.path.abspath(
            os.path.expanduser(
                self.config["execution"].pop("ads_config", DEFAULT_ADS_CONFIG_FOLDER)
            )
        )
        self._fill_exec_config_with_defaults(ads_config_path)
        # 4. fill in values from default infra config, if missing
        self._fill_ml_job_config_with_defaults(ads_config_path)
        if self.config["execution"].get("oci_config", None) and self.config[
            "execution"
        ].get("oci_profile", None):
            self._check_oci_config(
                self.config["execution"]["oci_config"],
                self.config["execution"]["oci_profile"],
            )
        logger.debug(f"Config: {self.config}")
        return self

    def _merge_config_with_cmd_args(self, cmd_args: Dict) -> None:
        # overwrite config with command line args
        # if a command line arg value is None or empty collection, then it is ignored
        def _merge(cfg, args):
            for k, v in cfg.items():
                if isinstance(v, dict):
                    _merge(v, args)
                elif args.get(k, None) and len(args[k]) > 0:
                    cfg[k] = args.pop(k)
                elif (
                    k in args
                    and not isinstance(args[k], bool)
                    and (args[k] is None or len(args[k]) == 0)
                ):
                    args.pop(k)

        _merge(self.config, cmd_args)
        if "execution" not in self.config:
            self.config["execution"] = {}
        for k, v in cmd_args.items():
            if not isinstance(v, bool):
                if v is not None and len(v) > 0:
                    self.config["execution"][k] = v
            else:
                self.config["execution"][k] = v

    def _fill_config_from_conf(self) -> None:
        if self.config["execution"].get("conf_file", None):
            conf_file = self.config["execution"]["conf_file"]
            conf_profile = self.config["execution"].get("conf_profile", DEFAULT_PROFILE)
            logger.info(f"Reading from {conf_profile} using profile {conf_profile}")
            parser = _read_from_ini(conf_file)
            extra_configs = dict(os.environ)
            extra_configs.update(parser[conf_profile])
            config_string = yaml.dump(self.config)
            # _DefaultNoneDict is used so that if $variable is not found in a section in .ini, None value is filled in.
            self.config = yaml.safe_load(
                Template(config_string).substitute(_DefaultNoneDict(**extra_configs))
            )

    def _fill_exec_config_with_defaults(self, ads_config_path: str) -> None:
        exec_config = self._get_default_exec_config()
        exec_config.update(
            {
                k: v
                for k, v in self._get_exec_config_from_conf(ads_config_path).items()
                if v is not None
            }
        )
        if not self.config.get("execution", None):
            self.config["execution"] = {}
        for k, v in exec_config.items():
            if v and not self.config["execution"].get(k, None):
                self.config["execution"][k] = v

    def _fill_ml_job_config_with_defaults(self, ads_config_path: str) -> None:
        # should be called after _fill_exec_config_with_defaults()
        oci_profile = self.config["execution"].get("oci_profile", DEFAULT_PROFILE)
        logger.info(f"Loading ML Job config for oci profile {oci_profile}.")
        infra_config = self._get_ml_job_config_from_conf(oci_profile, ads_config_path)
        if not self.config.get("infrastructure", None):
            self.config["infrastructure"] = {}
        if "conda_pack_os_prefix" not in self.config["execution"] and infra_config.get(
            "conda_pack_os_prefix", None
        ):
            self.config["execution"]["conda_pack_os_prefix"] = infra_config.pop(
                "conda_pack_os_prefix"
            )
        for k, v in infra_config.items():
            if v and not self.config["infrastructure"].get(k, None):
                self.config["infrastructure"][k] = v

    @staticmethod
    def _get_default_exec_config() -> Dict:
        return {
            "oci_config": DEFAULT_OCI_CONFIG_FILE,
            "oci_profile": DEFAULT_PROFILE,
            "conda_pack_folder": DEFAULT_CONDA_PACK_FOLDER,
        }

    @staticmethod
    def _get_exec_config_from_conf(ads_config_folder: str) -> Dict:
        # fill in oci and conda path
        if os.path.exists(os.path.join(ads_config_folder, "config.ini")):
            parser = _read_from_ini(os.path.join(ads_config_folder, "config.ini"))
            return {
                "oci_config": parser["OCI"].get("oci_config", None),
                "oci_profile": parser["OCI"].get("oci_profile", None),
                "conda_pack_folder": parser["CONDA"].get("conda_pack_folder", None),
                "conda_pack_os_prefix": parser["CONDA"].get(
                    "conda_pack_os_prefix", None
                ),
            }
        else:
            logger.info(
                f"{os.path.join(ads_config_folder, 'config.ini')} does not exist. No config loaded."
            )
            return {}

    @staticmethod
    def _get_ml_job_config_from_conf(oci_profile: str, ads_config_folder: str) -> Dict:
        # fill in ml job infra spec
        if os.path.exists(os.path.join(ads_config_folder, "ml_job_config.ini")):
            parser = _read_from_ini(
                os.path.join(ads_config_folder, "ml_job_config.ini")
            )
            if oci_profile in parser:
                return parser[oci_profile]
        else:
            logger.info(
                f"{os.path.join(ads_config_folder, 'ml_job_config.ini')} does not exist. No config loaded."
            )
        return {}

    @staticmethod
    def _check_oci_config(oci_config: str, oci_profile: str) -> None:
        parser = _read_from_ini(os.path.expanduser(oci_config))
        if oci_profile not in parser:
            raise ValueError(f"PROFILE {oci_profile} not found in {oci_config}.")
