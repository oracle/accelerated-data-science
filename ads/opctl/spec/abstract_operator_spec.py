#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from tempfile import gettempprefix
import fsspec
import yaml
import ads
import os
import oci
import json
from ads.jobs import Job
import ipaddress
from abc import ABCMeta, abstractmethod
import subprocess
from time import sleep, time
import pandas as pd  # Have to find a better way for timedelta
from urllib.parse import urlparse
from ads.opctl.utils import suppress_traceback


class AbstractOperatorSpec(metaclass=ABCMeta):
    YAML_TRANSLATION = {
        "docker_image": NotImplementedError,
    }

    def __init__(self, yaml_file, **kwargs):
        # TODO: Move into factory method, in order to determine OpSpec instance
        self.debug = kwargs["debug"]
        self.spec = _ingest_yaml_file(yaml_file, self.debug)
        if yaml_file:
            if os.path.exists(yaml_file):
                with open(yaml_file, "r") as f:
                    self.spec = suppress_traceback(self.debug)(yaml.safe_load)(f.read())
            else:
                raise FileNotFoundError(f"{yaml_file} is not found")
        else:
            self.spec = {}

        # TODO: init starts here:

        # Validate user yaml input
        # TODO: ask infra group: merge in additional args from cli, env vars, defaults?
        # Or should yaml files be forced to be completed, and not overwritten (
        # maybe there's lee-way? backend can change from cli or something?)
        # then re-validate?

        # Need todo equivalent of:
        # p = ConfigProcessor(config).step(ConfigMerger, **kwargs)

        # p.step(ConfigResolver).step(ConfigValidator)
        # # spec may have changed during validation step (e.g. defaults filled in)
        # # thus command need to be updated since it encodes spec
        # p = ConfigResolver(p.config)
        # p._resolve_command()
        # self.spec = p.config

    @staticmethod
    def build(yaml_file, **kwargs):
        return AbstractOperatorSpec(yaml_file, **kwargs)

    def __getitem__(self, key):
        # Note, we have not written a __setitem__ method because we do not want users setting these outside of the class
        return self._translate_key(key)

    def _translate_key(self, key):
        key_location = self.YAML_TRANSLATION[key]
        return self.spec.__getitem__(key_location)

    def get_keys(self):
        return self.YAML_TRANSLATION.keys()

    @property
    def docker_image(self):
        pass


class V0Spec(AbstractOperatorSpec):
    YAML_TRANSLATION = {
        "docker_image": ["execution"]["image"],
        "backend": ["execution"]["backend"],
        "operator_slug": ["execution"]["operator_slug"],
        "debug": ["execution"]["debug"],
        "operator_folder_path": ["execution"]["operator_folder_path"],
        "auth": ["execution"]["auth"],
        "oci_profile": ["execution"]["oci_profile"],
        "oci_config": ["execution"]["oci_config"],
        "conda_pack_folder": ["execution"]["conda_pack_folder"],
        "conda_pack_os_prefix": ["execution"]["conda_pack_os_prefix"],
        "compartment_id": ["infrastructure"]["compartment_id"],
        "project_id": ["infrastructure"]["project_id"],
        "subnet_id": ["infrastructure"]["subnet_id"],
        "log_group_id": ["infrastructure"]["log_group_id"],
        "log_id": ["infrastructure"]["log_id"],
        "shape_name": ["infrastructure"]["shape_name"],
        "block_storage_size_in_GBs": ["infrastructure"]["block_storage_size_in_GBs"],
        "docker_registry": ["infrastructure"]["docker_registry"],
    }

    @staticmethod
    def build(yaml_file, **kwargs):
        return V0Spec(yaml_file, **kwargs)


class DistributedV1Spec(AbstractOperatorSpec):
    # TODO: move this to a JobsClusterInfra spec class
    YAML_TRANSLATION = {
        # framework
        "framework_config_name": ["spec"]["Framework"]["spec"]["frameworkConfigName"],
        "framework_config": ["spec"]["Framework"]["spec"]["frameworkConfig"],
        "nanny_port_range": ["spec"]["Framework"]["spec"]["schedulerConfig"][
            "nannyPortRange"
        ],
        "worker_port_range": ["spec"]["Framework"]["spec"]["schedulerConfig"][
            "workerPortRange"
        ],
        "timeout": ["spec"]["Framework"]["spec"]["workerConfig"]["timeout"],
        "work_dir": ["spec"]["Framework"]["spec"]["workDir"],
        # Runtime
        "entrypoint": ["spec"]["Runtime"]["spec"]["entrypoint"],
        "args": ["spec"]["Runtime"]["spec"]["args"],
        "environmentVariables": ["spec"]["Runtime"]["spec"]["environmentVariables"],
        # Infra
        "docker_image": ["spec"]["Infrastructure"]["spec"]["dockerImage"],
        "display_name": ["spec"]["Infrastructure"]["spec"]["displayName"],
        "compartment_id": ["spec"]["Infrastructure"]["spec"]["compartmentId"],
        "project_id": ["spec"]["Infrastructure"]["spec"]["projectId"],
        "subnet_id": ["spec"]["Infrastructure"]["spec"]["subnetId"],
        "log_group_id": ["spec"]["Infrastructure"]["spec"]["logGroupId"],
        "log_id": ["spec"]["Infrastructure"]["spec"]["logId"],
        "shape_name": ["spec"]["Infrastructure"]["spec"]["cluster"]["shapeName"],
        "block_storage_size_in_GBs": ["spec"]["Infrastructure"]["spec"]["cluster"][
            "blockStorageSizeGB"
        ],
        "num_workers": ["spec"]["Infrastructure"]["spec"]["cluster"]["numWorkers"],
        "ephemeral": ["spec"]["Infrastructure"]["spec"]["cluster"]["ephemeral"],
    }

    @staticmethod
    def build(yaml_file, **kwargs):
        return DistributedV1Spec(yaml_file, **kwargs)


def _ingest_yaml_file(yaml_file, debug=False):
    if yaml_file:
        if os.path.exists(yaml_file):
            with open(yaml_file, "r") as f:
                spec = suppress_traceback(debug)(yaml.safe_load)(f.read())
        else:
            raise FileNotFoundError(f"{yaml_file} is not found")
    else:
        spec = {}
    return spec
