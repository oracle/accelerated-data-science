#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from abc import ABC, abstractmethod
from typing import Any


class Loader(ABC):
    """Data Loader Interface."""

    @abstractmethod
    def load(self, **kwargs) -> Any:
        pass
