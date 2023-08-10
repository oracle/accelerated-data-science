#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging
import os

logger = logging.getLogger(__name__)
from ads.common.oci_mixin import OCIModelMixin
import oci.feature_store
import yaml


try:
    from yaml import CDumper as dumper
    from yaml import CLoader as loader
except:
    from yaml import Dumper as dumper
    from yaml import Loader as loader

try:
    from odsc_cli.utils import user_fs_config_loc, FsTemplate
except ImportError:
    pass


class OCIFeatureStoreMixin(OCIModelMixin):
    __mod_time = 0
    __template: "FsTemplate" = None
    FS_SERVICE_ENDPOINT = "fs_service_endpoint"
    SERVICE_ENDPOINT = "service_endpoint"

    @classmethod
    def init_client(
        cls, **kwargs
    ) -> oci.feature_store.feature_store_client.FeatureStoreClient:
        default_kwargs: dict = cls._get_auth().get("client_kwargs", {})

        fs_service_endpoint = (
            kwargs.get(cls.FS_SERVICE_ENDPOINT, None)
            or kwargs.get(cls.SERVICE_ENDPOINT, None)
            or default_kwargs.get(cls.FS_SERVICE_ENDPOINT, None)
        )

        if not fs_service_endpoint:
            try:
                mod_time = os.stat(user_fs_config_loc()).st_mtime
                if mod_time > cls.__mod_time:
                    with open(user_fs_config_loc()) as ccf:
                        cls.__template = FsTemplate(yaml.load(ccf, Loader=loader))
                    cls.__mod_time = mod_time
            except NameError:
                logger.info(
                    "%s",
                    "Feature store configuration helpers are missing. "
                    "Support for reading service endpoint from config file is disabled",
                )
            except FileNotFoundError:
                logger.info(
                    "%s",
                    "ODSC cli config for feature store was not found",
                )
                pass
            if cls.__template:
                fs_service_endpoint = cls.__template.service_endpoint

        if fs_service_endpoint:
            kwargs[cls.SERVICE_ENDPOINT] = fs_service_endpoint

        client = cls._init_client(
            client=oci.feature_store.feature_store_client.FeatureStoreClient, **kwargs
        )
        return client

    @property
    def client(self) -> oci.feature_store.feature_store_client.FeatureStoreClient:
        return super().client
