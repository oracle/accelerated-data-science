#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
from typing import Dict
from ads.jobs.builders.infrastructure.dsc_job import DataScienceJob


class CustomScriptStep(DataScienceJob):
    def __init__(self, spec: Dict = None, **kwargs) -> None:
        """Initialize a custom script step infrastructure.

        Example
        -------
        Here is an example for defining a custom script step infrastructure using builder:

        .. code-block:: python

            from ads.pipeline import CustomScriptStep
            # Define an OCI Data Science custom script step infrastructure
            infrastructure = (
                CustomScriptStep()
                .with_shape_name("VM.Standard2.1")
                .with_block_storage_size(50)
            )

        See Also
        --------
        https://docs.oracle.com/en-us/iaas/tools/ads-sdk/latest/user_guide/pipeline/index.html
        """
        super().__init__(spec, **kwargs)
