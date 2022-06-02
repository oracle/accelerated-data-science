# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from abc import ABC, abstractmethod
from typing import Any, Generator


class Reader(ABC):
    """The Data Reader Interface."""

    @abstractmethod
    def read(self) -> Generator[Any, Any, Any]:
        """The abstract method to read data.

        Yields
        ------
        Generator[Any, Any, Any]
        """
        pass
