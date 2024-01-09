#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os

import oci.data_science

from ads.common.decorator.utils import class_or_instance_method
from ads.common.oci_mixin import OCIModelMixin

ENV_VAR_OCI_ODSC_SERVICE_ENDPOINT = "OCI_ODSC_SERVICE_ENDPOINT"


class OCIDataScienceMixin(OCIModelMixin):
    @class_or_instance_method
    def init_client(cls, **kwargs) -> oci.data_science.DataScienceClient:
        client_kwargs = kwargs.get("client_kwargs", {})
        if os.environ.get(ENV_VAR_OCI_ODSC_SERVICE_ENDPOINT):
            client_kwargs.update(
                dict(service_endpoint=os.environ.get(ENV_VAR_OCI_ODSC_SERVICE_ENDPOINT))
            )
            kwargs.update(client_kwargs)
        return cls._init_client(client=oci.data_science.DataScienceClient, **kwargs)

    @property
    def client(self) -> oci.data_science.DataScienceClient:
        return super().client

    @property
    def client_composite(self) -> oci.data_science.DataScienceClientCompositeOperations:
        return oci.data_science.DataScienceClientCompositeOperations(self.client)


class DSCNotebookSession(OCIDataScienceMixin, oci.data_science.models.NotebookSession):
    """Represents a data science notebook session

    To get the information of an existing notebook session:
    >>> notebook = DSCNotebookSession.from_ocid(NOTEBOOK_OCID)
    Get the name of the notebook session
    >>> notebook.display_name
    Get the subnet ID of the notebook session
    >>> notebook.notebook_session_configuration_details.subnet_id
    """
