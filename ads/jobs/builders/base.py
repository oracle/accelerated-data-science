#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import copy
from typing import Any, Dict, TypeVar, Type
from ads.jobs.serializer import Serializable


Self = TypeVar("Self", bound="Builder")
"""Special type to represent the current enclosed class.

This type is used by factory class method or when a method returns ``self``.
"""


class Builder(Serializable):
    attribute_map = {}

    def __init__(self, spec: Dict = None, **kwargs) -> None:
        """To initialize the object,
        user can either pass in the specification as a dictionary or through keyword arguments.

        Parameters
        ----------
        spec : dict, optional
            Object specification, by default None
        kwargs: dict
            Specification as keyword arguments.
            If spec contains the same key as the one in kwargs, the value from kwargs will be used.
        """
        super().__init__()
        self._spec = self._load_default_properties()
        self._spec.update(self._standardize_spec(spec))
        self._spec.update(self._standardize_spec(kwargs))

    def _load_default_properties(self):
        """
        Load default properties from environment variables, notebook session, etc.
        Should be implemented in the child classes.

        Returns
        -------
        Dict
            A dictionary of default properties.
        """
        return {}

    def _standardize_spec(self, spec) -> dict:
        if not spec:
            return {}
        snake_to_camel_map = {v: k for k, v in self.attribute_map.items()}
        for key in list(spec.keys()):
            if key not in self.attribute_map and key in snake_to_camel_map:
                spec[snake_to_camel_map[key]] = spec.pop(key)
        return spec

    def set_spec(self: Self, k: str, v: Any) -> Self:
        """Sets a specification property for the object.

        Parameters
        ----------
        k: str
            key, the name of the property.
        v: Any
            value, the value of the property.

        Returns
        -------
        Self
            This method returns self to support chaining methods.
        """
        if v is not None:
            self._spec[k] = v
        else:
            self._spec.pop(k, None)
        return self

    def get_spec(self, key: str, default: Any = None) -> Any:
        """Gets the value of a specification property

        Parameters
        ----------
        key : str
            The name of the property.
        default : Any, optional
            The default value to be used, if the property does not exist, by default None.

        Returns
        -------
        Any
            The value of the property.
        """
        return self._spec.get(key, default)

    @property
    def kind(self) -> str:
        """The kind of the object as showing in YAML.
        Subclass should overwrite this value.
        """
        return "builder"

    @property
    def type(self) -> str:
        """The type of the object as showing in YAML.

        This implementation returns the class name with the first letter coverted to lower case.
        """
        class_name = self.__class__.__name__
        return class_name[0].lower() + class_name[1:] if len(class_name) > 1 else ""

    def to_dict(self, **kwargs) -> dict:
        """Converts the object to dictionary with kind, type and spec as keys.

        Parameters
        ----------
        **kwargs: Dict
            The additional arguments.
            - filter_by_attribute_map: bool
                If True, then in the result will be included only the fields
                presented in the `attribute_map`.
        """
        filter_by_attribute_map = kwargs.pop("filter_by_attribute_map", False)

        spec = {
            key: value
            for key, value in copy.deepcopy(self._spec).items()
            if not filter_by_attribute_map or key in self.attribute_map
        }

        for key, value in spec.items():
            if hasattr(value, "to_dict"):
                value = value.to_dict()
            spec[key] = value

        return {
            "kind": self.kind,
            "type": self.type,
            "spec": spec,
        }

    @classmethod
    def from_dict(cls: Type[Self], obj_dict: dict) -> Self:
        """Initialize the object from a Python dictionary"""
        return cls(spec=obj_dict.get("spec"))

    def __repr__(self) -> str:
        """Displays the object as YAML."""
        return self.to_yaml()

    def build(self) -> Self:
        """Load default values from the environment for the job infrastructure.
        Should be implemented on the child level.
        """
        return self