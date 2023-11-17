#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.common.decorator.runtime_dependency import (
    OptionalDependency,
    runtime_dependency,
)

try:
    import scrubadub
except ImportError:
    raise ModuleNotFoundError(
        f"`scrubadub` module was not found. Please run "
        f"`pip install {OptionalDependency.PII}`."
    )


class EmailReplacer(scrubadub.post_processors.PostProcessor):
    name = "email_replacer"

    @runtime_dependency(module="faker", install_from=OptionalDependency.PII)
    def process_filth(self, filth_list):
        from faker import Faker

        for filth in filth_list:
            if filth.replacement_string:
                continue
            if filth.type.lower() != "email":
                continue
            filth.replacement_string = Faker().email()
        return filth_list
