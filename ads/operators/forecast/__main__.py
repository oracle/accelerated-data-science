#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import json
from .forecast import operate

args = json.loads(os.environ.get("OPERATOR_ARGS", "{}"))

operate(args)