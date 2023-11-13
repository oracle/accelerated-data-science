#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Sequence

from faker import Faker
from scrubadub.filth import Filth
from scrubadub.post_processors import PostProcessor


class EmailReplacer(PostProcessor):
    name = "email_replacer"

    def process_filth(self, filth_list: Sequence[Filth]) -> Sequence[Filth]:
        for filth in filth_list:
            if filth.replacement_string:
                continue
            if filth.type.lower() != "email":
                continue
            filth.replacement_string = Faker().email()
        return filth_list
