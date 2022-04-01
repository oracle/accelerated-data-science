#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from abc import ABC, abstractmethod
from typing import Any
from ads.common.serializer import Serializable


class Reader(ABC):
    """Data Reader Interface."""

    def info(self) -> Serializable:
        NotImplementedError(
            f"The class {self.__class__.__name__} did not implement the required method "
            "`info()`. Contact the class maintainer."
        )

    @abstractmethod
    def read(self) -> Any:
        pass
