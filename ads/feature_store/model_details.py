#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from copy import deepcopy
from typing import Dict, List

from ads.jobs.builders.base import Builder


class ModelDetails(Builder):
    """Sets the model Details.
    Methods
    -------
        with_items(self, items: List[str]) -> "ModelDetails"
        Set the model ids associated with a dataset
    """

    CONST_ITEMS = "items"

    attribute_map = {
        CONST_ITEMS: "items",
    }

    def __init__(self, items: List[str] = None) -> None:
        super().__init__()

        if items is None:
            items = []
        if items:
            self.with_items(items)

    @property
    def items(self) -> List[str]:
        return self.get_spec(self.CONST_ITEMS)

    @items.setter
    def items(self, items: List[str]):
        self.with_items(items)

    def with_items(self, items: List[str]) -> "ModelDetails":
        """Sets the model ids associated with dataset.

        Parameters
        ----------
        items: List[str]
            items array of model ids

        Returns
        -------
        ModelDetails
            The ModelDetails instance (self)
        """
        return self.set_spec(self.CONST_ITEMS, items)

    @property
    def kind(self) -> str:
        """The kind of the object as showing in a YAML."""
        return "model_details"

    def to_dict(self) -> Dict:
        """Serializes rule to a dictionary.

        Returns
        -------
        dict
            The rule resource serialized as a dictionary.
        """

        spec = deepcopy(self._spec)
        return spec
