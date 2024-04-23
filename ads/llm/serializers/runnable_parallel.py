#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from langchain.schema.runnable import RunnableParallel
from langchain.load.dump import dumpd
from langchain.load.load import load


class RunnableParallelSerializer:
    @staticmethod
    def type():
        return RunnableParallel.__name__

    @staticmethod
    def load(config: dict, **kwargs):
        steps = config.get("kwargs", dict()).get("steps", dict())
        steps = {k: load(v, **kwargs) for k, v in steps.items()}
        return RunnableParallel(**steps)

    @staticmethod
    def save(obj):
        serialized = dumpd(obj)
        serialized["_type"] = RunnableParallelSerializer.type()
        return serialized
