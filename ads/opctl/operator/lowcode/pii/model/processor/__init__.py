#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from .email_replacer import EmailReplacer
from .mbi_replacer import MBIReplacer
from .name_replacer import NameReplacer
from .number_replacer import NumberReplacer
from .remover import Remover

POSTPROCESSOR_MAP = {
    item.name.lower(): item
    for item in [
        NameReplacer,
        NumberReplacer,
        EmailReplacer,
        MBIReplacer,
        Remover,
    ]
}

# Currently only support anonymization for the following entity.
SUPPORTED_REPLACER = {
    "name": NameReplacer,
    "number": NumberReplacer,
    "phone": NumberReplacer,
    "social_security_number": NumberReplacer,
    "fin": NumberReplacer,
    "mrn": NumberReplacer,
    "email": EmailReplacer,
    "mbi": MBIReplacer,
}
