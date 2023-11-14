#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import random
import string

from ads.common.decorator.runtime_dependency import OptionalDependency

try:
    import scrubadub
except ImportError:
    raise ModuleNotFoundError(
        f"`scrubadub` module was not found. Please run "
        f"`pip install {OptionalDependency.PII}`."
    )


class MBIReplacer(scrubadub.post_processors.PostProcessor):
    name = "mbi_replacer"
    CHAR_POOL = "ACDEFGHJKMNPQRTUVWXY"

    def generate_mbi(self):
        return "".join(random.choices(self.CHAR_POOL + string.digits, k=11))

    def process_filth(self, filth_list):
        for filth in filth_list:
            if filth.replacement_string:
                continue
            if filth.type.lower() != "mbi":
                continue
            filth.replacement_string = self.generate_mbi()
        return filth_list
