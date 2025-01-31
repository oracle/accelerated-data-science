#!/usr/bin/env python
# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from dataclasses import dataclass

from ads.common.serializer import DataClassSerializable


@dataclass(repr=False)
class AquaResourceIdentifier(DataClassSerializable):
    id: str = ""
    name: str = ""
    url: str = ""
