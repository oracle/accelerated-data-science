#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import re
from typing import Callable


class UDF:
    @staticmethod
    def from_regex(regex: str) -> Callable:
        def function(content):
            match = re.match(regex, content)
            if match:
                if len(match.groups()) == 0:
                    return [match.group(0)]
                else:
                    return match.groups()

        return function
