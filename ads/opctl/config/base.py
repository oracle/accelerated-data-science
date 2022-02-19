#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from abc import abstractmethod
from typing import Dict


class ConfigProcessor:
    def __init__(self, config: Dict = None) -> None:
        """
        Initializing a ConfigProcessor object given a configuration.

        Parameters
        ----------
        config: dict
            a dictionary of configurations
        """
        self.config = config if config else {}

    @abstractmethod
    def process(self, **kwargs) -> "ConfigProcessor":
        """
        Perform some processing on configuration
        associated with this object.

        Parameters
        ----------
        kwargs: dict
            keyword arguments passed to the function

        Returns
        -------
        ConfigProcessor
            this instance itself
        """
        pass

    def step(self, cls, **kwargs) -> "ConfigProcessor":
        """
        Perform some processing according to the cls given.

        Parameters
        ----------
        cls: `ConfigProcessor`
            a subclass of `ConfigProcessor`
        kwargs: dict
            keyword arguments passed to the `process` function of `cls`

        Returns
        -------
        `ConfigProcessor`
            this instance itself
        """
        return cls(self.config).process(**kwargs)
