#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import functools
import os


class NotSupportedError(Exception):   # pragma: no cover
    pass


def experimental(cls):
    @functools.wraps(cls)
    def wrapper(*args, **kwargs):
        instance = cls(*args, **kwargs)
        print(f"{cls.__name__} is experimental and may be removed in the future.")
        return instance

    return wrapper


class PY4JGateway:
    def __init__(self) -> None:
        try:
            from py4j.java_gateway import GatewayParameters, JavaGateway, launch_gateway
        except ModuleNotFoundError:
            raise ModuleNotFoundError("py4j is not installed.")
        if "CONDA_PREFIX" not in os.environ or not os.path.exists(
            os.path.join(os.environ.get("CONDA_PREFIX"), "text-extraction-tools.jar")
        ):
            raise NotSupportedError(
                "Tika is not supported in this distribution. Use alternatives such as pdfplumber."
            )
        port = launch_gateway(
            java_path="/usr/bin/java",
            classpath=os.path.join(
                os.environ.get("CONDA_PREFIX"), "text-extraction-tools.jar"
            ),
        )
        self.gateway = JavaGateway(gateway_parameters=GatewayParameters(port=port))

    def __enter__(self) -> None:
        return self.gateway

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.gateway.shutdown()
