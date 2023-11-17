#!/usr/bin/env python
# -*- coding: utf-8; -*-
from types import MethodType

from ads.common.decorator.utils import class_or_instance_method
from oci.signer import AbstractBaseSigner

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging
import email.utils
import os
import oci
import feature_store_client.feature_store as fs

logger = logging.getLogger(__name__)
from ads.common.oci_mixin import OCIModelMixin
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
    def init_client(cls, **kwargs) -> fs.feature_store_client.FeatureStoreClient:
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

        client: fs.FeatureStoreClient = cls._init_client(
            client=fs.FeatureStoreClient, **kwargs
        )
        signer: oci.Signer = client.base_client.signer
        signer.do_request_sign = MethodType(fs_do_request_sign, signer)
        return client

    @property
    def client(self) -> fs.feature_store_client.FeatureStoreClient:
        return super().client

    @class_or_instance_method
    def list_resource(
        cls, compartment_id: str = None, limit: int = 0, **kwargs
    ) -> list:
        """Generic method to list OCI resources

        Parameters
        ----------
        compartment_id : str
            Compartment ID of the OCI resources. Defaults to None.
            If compartment_id is not specified,
            the value of NB_SESSION_COMPARTMENT_OCID in environment variable will be used.
        limit : int
            The maximum number of items to return. Defaults to 0, All items will be returned
        **kwargs :
            Additional keyword arguments to filter the resource.
            The kwargs are passed into OCI API.

        Returns
        -------
        list
            A list of OCI resources

        Raises
        ------
        NotImplementedError
            List method is not supported or implemented.

        """
        if limit:
            items = cls._find_oci_method("list")(
                cls.check_compartment_id(compartment_id), limit=limit, **kwargs
            ).data.items
        else:
            items = oci.pagination.list_call_get_all_results(
                cls._find_oci_method("list"),
                cls.check_compartment_id(compartment_id),
                **kwargs,
            ).data
        return [cls.from_oci_model(item) for item in items]


def inject_missing_headers(request):
    # Inject date, host, and content-type if missing
    date = email.utils.formatdate(usegmt=True)
    if request.path_url.startswith("/20230101"):
        request.headers.setdefault("x-date", date)
        request.headers.setdefault(
            "path", request.method.lower() + " " + request.path_url
        )
    request.headers.setdefault("date", date)


def fs_do_request_sign(self, request, enforce_content_headers=True):
    inject_missing_headers(request)
    do_request_sign = MethodType(AbstractBaseSigner.do_request_sign, self)
    return do_request_sign(request, enforce_content_headers)

    # inject_missing_headers_og(request, sign_body, enforce_content_headers)
