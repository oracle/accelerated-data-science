#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


class InvalidFeatureType(TypeError):
    def __init__(self, tname: str):
        super().__init__(f"Type {tname} is not a sublcass of FeatureType.")


class TypeAlreadyRegistered(TypeError):
    def __init__(self, tname: str):
        super().__init__(f"Type {tname} already registered.")


class TypeAlreadyAdded(TypeError):
    def __init__(self, tname: str):
        super().__init__(f"Type {tname} is already added.")


class TypeNotFound(TypeError):
    def __init__(self, tname: str):
        super().__init__(f"Type {tname} is not found.")


class NameAlreadyRegistered(NameError):
    def __init__(self, name: str):
        super().__init__(f"Type with name {name} already registered.")


class WarningAlreadyExists(ValueError):
    def __init__(self, name: str):
        super().__init__(f"Warning {name} already exists.")


class WarningNotFound(ValueError):
    def __init__(self, name: str):
        super().__init__(f"Warning {name} is not found.")
