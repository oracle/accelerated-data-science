#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
from ads.common.decorator.runtime_dependency import OptionalDependency

try:
    import scrubadub
except ImportError:
    raise ModuleNotFoundError(
        f"`scrubadub` module was not found. Please run "
        f"`pip install {OptionalDependency.PII}`."
    )


class Remover(scrubadub.post_processors.PostProcessor):
    name = "remover"
    _ENTITIES = []

    def process_filth(self, filth_list):
        for filth in filth_list:
            if filth.type.lower() in self._ENTITIES:
                filth.replacement_string = ""

        return filth_list
