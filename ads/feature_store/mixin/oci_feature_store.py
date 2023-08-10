#!/usr/bin/env python
# -*- coding: utf-8; -*-
from ads.common.decorator.utils import class_or_instance_method

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.common.oci_mixin import OCIModelMixin
import oci.feature_store
import os


class OCIFeatureStoreMixin(OCIModelMixin):
    @classmethod
    def init_client(
        cls, **kwargs
    ) -> oci.feature_store.feature_store_client.FeatureStoreClient:
        # TODO: Getting the endpoint from authorizer
        fs_service_endpoint = os.environ.get("OCI_FS_SERVICE_ENDPOINT")
        if fs_service_endpoint:
            kwargs = {"service_endpoint": fs_service_endpoint}

        client = cls._init_client(
            client=oci.feature_store.feature_store_client.FeatureStoreClient, **kwargs
        )
        return client

    @property
    def client(self) -> oci.feature_store.feature_store_client.FeatureStoreClient:
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
