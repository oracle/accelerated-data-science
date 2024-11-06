#!/usr/bin/env python

# Copyright (c) 2021, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import oci.functions

from ads.common.decorator.utils import class_or_instance_method
from ads.common.oci_mixin import OCIClientMixin, OCIModelMixin


class OCIFunctionsManagementMixin(OCIModelMixin):
    @class_or_instance_method
    def init_client(cls, **kwargs) -> oci.functions.FunctionsManagementClient:
        return cls._init_client(
            client=oci.functions.FunctionsManagementClient, **kwargs
        )

    @property
    def client(self) -> oci.functions.FunctionsManagementClient:
        return super().client

    @property
    def client_composite(
        self,
    ) -> oci.functions.FunctionsManagementClientCompositeOperations:
        return oci.functions.FunctionsManagementClientCompositeOperations(self.client)


class OCIFunctionsInvoke(OCIClientMixin):
    @class_or_instance_method
    def init_client(cls, **kwargs) -> oci.functions.FunctionsInvokeClient:
        return cls._init_client(client=oci.functions.FunctionsInvokeClient, **kwargs)

    @property
    def client(self) -> oci.functions.FunctionsInvokeClient:
        return super().client

    @property
    def client_composite(
        self,
    ) -> oci.functions.FunctionsInvokeClientCompositeOperations:
        return oci.functions.FunctionsInvokeClientCompositeOperations(self.client)
