#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.llm.langchain.inference_backend.const import InferenceFramework


class UnsupportedInferenceFrameworkError(Exception):
    def __init__(
        self,
        protocol: str,
    ):
        self.protocol = protocol
        super().__init__(
            (
                f"Unsupported inference protocol provided: {protocol}. "
                f"Supported protocols: {', '.join(InferenceFramework.values())}."
            )
        )
