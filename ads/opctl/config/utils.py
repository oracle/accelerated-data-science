#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import configparser
import os


class OperatorNotFound(Exception):
    pass


class CondaPackInfoNotProvided(Exception):
    pass


class NotSupportedError(Exception):
    pass


class _DefaultNoneDict(dict):
    def __missing__(self, key):
        return ""


def _read_from_ini(path: str) -> configparser.ConfigParser:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} is not found.")
    parser = configparser.ConfigParser(default_section=None)
    parser.optionxform = str  # preserve case
    parser.read(path)
    return parser
