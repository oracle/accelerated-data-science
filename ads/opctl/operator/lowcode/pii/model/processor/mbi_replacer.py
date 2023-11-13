#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import random
import string
from typing import Sequence

from scrubadub.filth import Filth
from scrubadub.post_processors import PostProcessor


class MBIReplacer(PostProcessor):
    name = "mbi_replacer"
    CHAR_POOL = "ACDEFGHJKMNPQRTUVWXY"

    def generate_mbi(self):
        return "".join(random.choices(self.CHAR_POOL + string.digits, k=11))

    def process_filth(self, filth_list: Sequence[Filth]) -> Sequence[Filth]:
        for filth in filth_list:
            if filth.replacement_string:
                continue
            if filth.type.lower() != "mbi":
                continue
            filth.replacement_string = self.generate_mbi()
        return filth_list
